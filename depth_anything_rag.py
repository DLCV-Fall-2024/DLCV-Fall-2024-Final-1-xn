import gc
import json
from PIL import Image
from pathlib import Path
import re
from typing import Dict, List

import faiss
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from depth_anything.dpt import DepthAnything
from feature_extractor import FeatureExtractor
from task_retriever import TaskRetriever


class RAGInference:
    def __init__(
        self,
        llava_model_id: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
        depth_model_id: str = "LiheYoung/depth_anything_vitl14",
        retrieval_indexes_dir: str = "retrieval_indexes",
        data_root: str = "data",
        output_dir: str = "rag_results",
        device: str = "cuda",
        k_examples: int = 2  # Reduced from 3 to 2
    ):
        self.device = device
        self.data_root = data_root
        self.output_dir = output_dir
        self.k_examples = k_examples
        os.makedirs(output_dir, exist_ok=True)

        print("Loading models...")
        self.depth_model = DepthAnything.from_pretrained(depth_model_id).to(device)
        self.depth_model.eval()
        self.feature_extractor = FeatureExtractor(self.depth_model, device)
        
        self.processor = LlavaNextProcessor.from_pretrained(llava_model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            llava_model_id, torch_dtype=torch.float16, device_map=device
        )
        
        self.retrievers = self._load_retrievers(retrieval_indexes_dir)
        self.all_results = {}

    def _load_retrievers(self, index_dir):
        retrievers = {}
        task_types = ["general_perception", "region_perception", "driving_suggestion"]

        for task_type in task_types:
            print(f"Loading {task_type} retriever...")
            task_dir = Path(index_dir) / task_type
            retriever = TaskRetriever(
                task_type=task_type,
                embedding_dim=self.feature_extractor.get_embedding_dim(),
                device=self.device,
            )
            retriever.load_index(task_dir)
            retrievers[task_type] = retriever

        return retrievers

    def get_prompt(self, task_type: str, retrieved_examples: List[Dict] = None) -> str:
        base_prompts = {
            "general_perception": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "recognizing traffic scenes and making detailed explanation. The expert receives an "
                "image of traffic captured from the perspective of the ego car. USER: <image>\n "
                "Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, "
                "buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), "
                "traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), "
                "traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not "
                "discuss any objects beyond the seven categories above. Please describe each object's "
                "color, position, status, implication, respones, and how they influence ego car. EXPERT:"
            ),
            "region_perception": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "recognizing traffic scenes and making detailed explanation. The expert receives an "
                "image of traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please describe the object inside the red rectangle in the image and explain why it "
                "affect ego car driving. EXPERT:"
            ),
            "driving_suggestion": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "providing specific and helpful driving suggestions. The expert receives an image of "
                "traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please provide driving suggestions for the ego car based on the current scene. EXPERT:"
            )
        }

        if not retrieved_examples:
            return base_prompts[task_type]
        
        prompt = "Brief similar examples:\n\n"
        for i, example in enumerate(retrieved_examples[:2]):
            # Truncate example answers to keep them short
            print(f"\nExample {i+1}:")
            print(f"Image: {example['id']}")
            if 'question' in example:
                print(f"Question: {example['question']}")
            if 'answer' in example:
                print(f"Answer: {example['answer']}")
                
            answer = example['answer']
            if len(answer) > 100:
                answer = answer[:100] + "..."
            prompt += f"Example {i+1}: {answer}\n\n"
            
        prompt += "\nBased on these examples, " + base_prompts[task_type]
        return prompt

    def post_process_output(self, result: str, task_type: str) -> str:
        """
        Post-process the model output to ensure completeness and proper formatting
        while handling truncation.
        """
        result = result.strip()
        
        # Split into sentences using multiple delimiters
        sentences = re.split('[.!?]', result)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return result
            
        if task_type == "driving_suggestion":
            # Handle numbered/bulleted lists
            if any(s.strip().startswith(('1.', '2.', '•', '-')) for s in sentences):
                processed_points = []
                current_point = []
                
                for s in sentences[:-1]:  # Exclude last potentially incomplete sentence
                    s = s.strip()
                    if s.startswith(('1.', '2.', '•', '-')):
                        if current_point:
                            processed_points.append(' '.join(current_point) + '.')
                            current_point = []
                        current_point.append(s)
                    else:
                        current_point.append(s)
                
                if current_point:
                    processed_points.append(' '.join(current_point) + '.')
                
                return ' '.join(processed_points)
            
        # For general and region perception
        complete_sentences = []
        current_sentence = []
        
        for s in sentences[:-1]:  # Exclude last potentially incomplete sentence
            s = s.strip()
            if s:
                current_sentence.append(s)
                
                # Check if this makes a complete thought
                combined = ' '.join(current_sentence)
                if len(combined.split()) >= 3:  # Minimum words for a complete thought
                    complete_sentences.append(combined + '.')
                    current_sentence = []
        
        # If we have complete sentences, use them
        if complete_sentences:
            final_result = ' '.join(complete_sentences)
        else:
            # If no complete sentences, use the original but ensure proper ending
            final_result = result.rstrip(',.!? ') + '.'
            
        return final_result

    def process_single_example(self, example: Dict, task_type: str):
        try:
            image_name = os.path.basename(example['image'])
            image_path = os.path.join(self.data_root, "images", image_name)
            image = Image.open(image_path)
            
            image_tensor = self.feature_extractor.preprocess_image(image).unsqueeze(0).to(self.device)
            features = self.feature_extractor.extract_and_concatenate(image_tensor)
            retrieved_examples, _ = self.retrievers[task_type].retrieve(features, k=self.k_examples)
            
            prompt = self.get_prompt(task_type, retrieved_examples)
            
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    use_cache=True
                )

            result = self.processor.decode(output[0], skip_special_tokens=True)
            result = result.split("EXPERT:", 1)[1] if "EXPERT:" in result else result
            
            # Post-process the result
            final_result = self.post_process_output(result.strip(), task_type)
            
            return {
                "question_id": example['id'],
                "answer": final_result
            }
            
        except Exception as e:
            print(f"Error processing example {example['id']}: {str(e)}")
            return None

    def process_all_tasks(self):
        tasks = [
            ("general_perception", "general_perception"),
            ("region_perception", "region_perception"),
            ("driving_suggestion", "driving_suggestion"),
        ]

        for task_type, filename in tasks:
            print(f"\nProcessing {task_type}...")
            data_file = os.path.join(
                self.data_root, "annotations", f"test_{filename}.jsonl"
            )

            try:
                with open(data_file, "r") as f:
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
            
        self.save_results(intermediate=False)

    def save_results(self, intermediate: bool = False):
        save_path = os.path.join(
            self.output_dir,
            "results_intermediate.json" if intermediate else "submission.json",
        )

        with open(save_path, "w") as f:
            json.dump(self.all_results, f, indent=2)

        if not intermediate:
            print(f"Final results saved to {save_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--llava_model_id', type=str, default="llava-hf/llava-v1.6-vicuna-7b-hf")
    parser.add_argument('--depth_model_id', type=str, default="LiheYoung/depth_anything_vitl14")
    parser.add_argument('--retrieval_indexes_dir', type=str, default="retrieval_indexes")
    parser.add_argument('--data_root', type=str, default="data")
    parser.add_argument('--output_dir', type=str, default="rag_results")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--k_examples', type=int, default=2)
    
    args = parser.parse_args()

    processor = RAGInference(
        llava_model_id=args.llava_model_id,
        depth_model_id=args.depth_model_id,
        retrieval_indexes_dir=args.retrieval_indexes_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        k_examples=args.k_examples,
    )

    processor.process_all_tasks()


if __name__ == "__main__":
    main()
