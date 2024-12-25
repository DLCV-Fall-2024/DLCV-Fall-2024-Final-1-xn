import json
import os
import shutil

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

# Configuration
MAX_TOKEN = 300
BATCH_SIZE = 8  # Adjust based on your GPU memory
OUTPUT_DIR = "inference_results"
FINE_TUNED_MODEL_DIR = "/content/dpo_results/checkpoint-45"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATA_ROOT = "data"

torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def clean_response(text):
    """Clean and format the model output."""
    # Split by EXPERT: if present
    if "EXPERT: " in text:
        text = text.split("EXPERT: ", 1)[1]

    # Find the last complete sentence
    last_period_idx = text.rfind(".")
    if last_period_idx == -1:
        # If no period found, add one at the end
        return text.strip() + "."

    # Keep all complete sentences
    cleaned_text = text[: last_period_idx + 1].strip()
    return cleaned_text


class DataProcessor:
    def __init__(self):
        self.setup_model()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.all_results = self.load_existing_results()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_existing_results(self):
        submission_path = os.path.join(OUTPUT_DIR, "submission.json")
        if os.path.exists(submission_path):
            with open(submission_path, "r") as f:
                return json.load(f)
        return {}

    def save_results(self, intermediate=True):
        if intermediate:
            save_path = os.path.join(OUTPUT_DIR, "submission_intermediate.json")
        else:
            save_path = os.path.join(OUTPUT_DIR, "submission.json")

        with open(save_path, "w") as f:
            json.dump(self.all_results, f, indent=2)

        if not intermediate:
            print(f"Final results saved to {save_path}")

    def setup_model(self):
        print("Setting up model...")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(
                f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB"
            )

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            FINE_TUNED_MODEL_DIR,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="cuda:0",
        )
        # self.model = PeftModel.from_pretrained(
        #     self.model, FINE_TUNED_MODEL_DIR, torch_dtype=torch.float16
        # )
        self.model.eval()

    def get_prompt(self, task_type):
        prompts = {
            "general": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "recognizing traffic scenes and making detailed explanation. The expert receives an "
                "image of traffic captured from the perspective of the ego car. USER: <image>\n "
                "Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, "
                "buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), "
                "traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), "
                "traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not "
                "discuss any objects beyond the seven categories above. Please describe each object's "
                "color, position, status, implication, responses, and how they influence ego car. EXPERT:"
            ),
            "region": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "recognizing traffic scenes and making detailed explanation. The expert receives an "
                "image of traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please describe the object inside the red rectangle in the image and explain why it "
                "affect ego car driving. EXPERT:"
            ),
            "driving": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "providing specific and helpful driving suggestions. The expert receives an image of "
                "traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please provide driving suggestions for the ego car based on the current scene. EXPERT:"
            ),
        }
        return prompts.get(task_type, "")

    def process_batch(self, batch_examples, task_type):
        """Process a batch of examples together."""
        # Skip already processed examples
        batch_examples = [
            ex for ex in batch_examples if ex["id"] not in self.all_results
        ]
        if not batch_examples:
            return []

        try:
            # Prepare batch inputs
            images = [Image.open(ex["image"]) for ex in batch_examples]
            prompt = self.get_prompt(task_type)
            prompts = [prompt] * len(images)

            # Process batch
            inputs = self.processor(
                text=prompts, images=images, return_tensors="pt", padding=True
            )
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # Generate outputs
            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs = self.model.generate(
                    **inputs, max_new_tokens=MAX_TOKEN, do_sample=False, use_cache=True
                )

            # Process results
            results = []
            for i, output in enumerate(outputs):
                result = self.processor.decode(output, skip_special_tokens=True)
                result = clean_response(result)  # Clean and format the response
                results.append(
                    {"question_id": batch_examples[i]["id"], "answer": result}
                )

            # Clean up
            del inputs, outputs
            torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"Error processing batch: {e}")
            return []

    def process_all_tasks(self):
        """Process all tasks with batching."""
        tasks = [
            ("general", "general_perception"),
            ("region", "region_perception"),
            ("driving", "driving_suggestion"),
        ]

        for task_type, filename in tasks:
            print(f"\nProcessing {task_type} task...")
            data_file = os.path.join(DATA_ROOT, "annotations", f"test_{filename}.jsonl")

            try:
                with open(data_file, "r") as f:
                    examples = [json.loads(line) for line in f]
            except FileNotFoundError:
                print(f"Error: Could not find file {data_file}")
                continue

            # Process in batches
            for i in tqdm(range(0, len(examples), BATCH_SIZE)):
                batch = examples[i : i + BATCH_SIZE]
                results = self.process_batch(batch, task_type)

                # Save results
                for result in results:
                    self.all_results[result["question_id"]] = result["answer"]

                if i % (BATCH_SIZE * 2) == 0:
                    self.save_results(intermediate=True)

            print(f"Completed {task_type} task, processed {len(examples)} examples")

        self.save_results(intermediate=False)
        create_submission_zip()


def create_submission_zip():
    api_key_path = os.path.join(OUTPUT_DIR, "api_key.txt")
    with open(api_key_path, "w") as f:
        f.write("YOUR_GEMINI_API_KEY")

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copy2(os.path.join(OUTPUT_DIR, "submission.json"), temp_dir)
        shutil.copy2(api_key_path, temp_dir)
        zip_path = os.path.join(OUTPUT_DIR, "pred")
        shutil.make_archive(zip_path, "zip", temp_dir)

    print(f"Submission zip created at {zip_path}.zip")


def main():
    try:
        processor = DataProcessor()
        processor.process_all_tasks()
        print("\nInference complete!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
