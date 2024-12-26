import json
import os
import shutil
import torch
import argparse
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from depth_anything.dpt import DepthAnything
from feature_extractor import FeatureExtractor
from task_retriever import TaskRetriever
from pathlib import Path
import gc

# Configuration
MAX_TOKEN = 300
BATCH_SIZE = 8
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEPTH_MODEL_ID = "LiheYoung/depth_anything_vitl14"
RETRIEVAL_INDEXES_DIR = "retrieval_indexes"
K_EXAMPLES = 2

def parse_args():
    parser = argparse.ArgumentParser(description='Run RAG-enhanced inference with LLaVA model')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                      help='Directory containing the fine-tuned model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save inference results')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory containing the data')
    return parser.parse_args()

torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def clean_response(text):
    """Clean and format the model output."""
    if "ASSISTANT: " in text:
        text = text.split("ASSISTANT: ", 1)[1]
        
    last_period_idx = text.rfind('.')
    if last_period_idx == -1:
        return text.strip()
    
    cleaned_text = text[:last_period_idx + 1].strip()
    return cleaned_text

class DataProcessor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.setup_models()
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.all_results = self.load_existing_results()
        self.retrieval_examples = {}

    def load_existing_results(self):
        submission_path = os.path.join(self.args.output_dir, "submission.json")
        if os.path.exists(submission_path):
            with open(submission_path, "r") as f:
                return json.load(f)
        return {}

    def setup_models(self):
        print("Setting up models...")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB")

        # Setup LLaVA model
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="cuda:0",
        )
        self.model = PeftModel.from_pretrained(
            self.model, self.args.ckpt_dir, torch_dtype=torch.float16
        )
        self.model.eval()

        # Setup Depth model and feature extractor
        self.depth_model = DepthAnything.from_pretrained(DEPTH_MODEL_ID).to(self.device)
        self.depth_model.eval()
        self.feature_extractor = FeatureExtractor(self.depth_model, self.device)

        # Setup retrievers
        self.retrievers = self._load_retrievers()

    def _load_retrievers(self):
        retrievers = {}
        task_types = ["general_perception", "region_perception", "driving_suggestion"]

        for task_type in task_types:
            print(f"Loading {task_type} retriever...")
            task_dir = Path(RETRIEVAL_INDEXES_DIR) / task_type
            retriever = TaskRetriever(
                task_type=task_type,
                embedding_dim=self.feature_extractor.get_embedding_dim(),
                device=self.device,
            )
            retriever.load_index(task_dir)
            retrievers[task_type] = retriever

        return retrievers

    def get_prompt(self, task_type, retrieved_examples=None):
        base_prompts = {
            "general_perception": (
                "Focus on objects influencing the ego car's driving behavior: "
                "vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), "
                "traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), "
                "traffic cones, barriers, and miscellaneous (debris, dustbin, animals, etc.). "
                "Do not discuss objects beyond these seven categories. "
                "For each object, describe its color, position, status, implications, and influence on the ego car."
            ),
            "region_perception": (
                "Describe the object inside the red rectangle in the image and explain why it affects the ego car's driving."
            ),
            "driving_suggestion": (
                "Provide driving suggestions for the ego car based on the current scene."
            ),
        }

        if not retrieved_examples:
            return base_prompts.get(task_type, "")

        prompt = (
            "Here are some example analyses to guide you. "
            "Please analyze the given image in a similar manner:\n\n"
        )

        for i, example in enumerate(retrieved_examples[:K_EXAMPLES]):
            prompt += f"Example {i+1}: {example.get('answer', '')}\n\n"

        prompt += "Now, analyze this image following the same style:\n"
        prompt += base_prompts.get(task_type, "")

        return prompt

    def process_single_example(self, example, task_type):
        image_name = os.path.basename(example['image'])
        image_path = os.path.join(self.args.data_root, "images", image_name)
        image = Image.open(image_path).convert('RGB')

        # Get similar examples using RAG
        image_tensor = self.feature_extractor.preprocess_image(image).unsqueeze(0).to(self.device)
        features = self.feature_extractor.extract_and_concatenate(image_tensor)
        retrieved_examples, _ = self.retrievers[task_type].retrieve(features, k=K_EXAMPLES)

        # Save retrieved examples
        self.save_retrieval_examples(example['id'], task_type, retrieved_examples)

        # Generate enhanced prompt with retrieved examples
        prompt_text = self.get_prompt(task_type, retrieved_examples)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process the current example
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            do_rescale=True,
            do_resize=True
        ).to(self.device)

        with torch.cuda.amp.autocast():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKEN,
                do_sample=False
            )

        result = self.processor.decode(output[0], skip_special_tokens=True)
        result = clean_response(result)

        del inputs, output
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "question_id": example['id'],
            "answer": result
        }

    def save_retrieval_examples(self, example_id, task_type, retrieved_examples, intermediate=False):
        examples_data = {
            "task_type": task_type,
            "retrieved_examples": [
                {
                    "id": ex["id"],
                    "answer": ex["answer"]
                } for ex in retrieved_examples[:K_EXAMPLES]
            ]
        }
        
        self.retrieval_examples[example_id] = examples_data
        
        output_file = os.path.join(
            self.args.output_dir, 
            "retrieval_examples_intermediate.json" if intermediate else "retrieval_examples.json"
        )
        
        with open(output_file, "w") as f:
            json.dump(self.retrieval_examples, f, indent=2)

    def process_all_tasks(self):
        tasks = [
            ("general_perception", "general_perception"),
            ("region_perception", "region_perception"),
            ("driving_suggestion", "driving_suggestion")
        ]

        for task_type, filename in tasks:
            print(f"\nProcessing {task_type}...")
            data_file = os.path.join(self.args.data_root, "annotations", f"test_{filename}.jsonl")

            try:
                with open(data_file, 'r') as f:
                    examples = [json.loads(line) for line in f]
            except FileNotFoundError:
                print(f"Error: Could not find file {data_file}")
                continue

            for i, example in enumerate(tqdm(examples)):
                result = self.process_single_example(example, task_type)
                if result:
                    self.all_results[result["question_id"]] = result["answer"]

                if i % 5 == 0:
                    self.save_results(intermediate=True)
                    self.save_retrieval_examples(example['id'], task_type, [], intermediate=True)

        self.save_results(intermediate=False)
        self.create_submission_zip()

    def save_results(self, intermediate=True):
        if intermediate:
            save_path = os.path.join(self.args.output_dir, "submission_intermediate.json")
        else:
            save_path = os.path.join(self.args.output_dir, "submission.json")
        
        with open(save_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        if not intermediate:
            print(f"Final results saved to {save_path}")

    def create_submission_zip(self):
        api_key_path = os.path.join(self.args.output_dir, "api_key.txt")
        with open(api_key_path, 'w') as f:
            f.write("YOUR_GEMINI_API_KEY")

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copy2(os.path.join(self.args.output_dir, "submission.json"), temp_dir)
            shutil.copy2(api_key_path, temp_dir)
            zip_path = os.path.join(self.args.output_dir, "pred")
            shutil.make_archive(zip_path, "zip", temp_dir)

        print(f"Submission zip created at {zip_path}.zip")

def main():
    args = parse_args()
    try:
        processor = DataProcessor(args)
        processor.process_all_tasks()
        print("\nProcessing complete!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()