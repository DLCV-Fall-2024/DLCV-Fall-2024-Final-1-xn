import gc
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from build_retriever_gdino import GDINOFeatureExtractor

from groundingdino.util.inference import load_model, load_image, predict
from task_retriever import TaskRetriever

class GDINORAGInference:
    def __init__(
        self,
        llava_model_id: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
        gdino_model_id: str = "path/to/grounding_dino_model",
        retrieval_indexes_dir: str = "retrieval_indexes",
        data_root: str = "data",
        output_dir: str = "rag_results",
        device: str = "cuda",
        k_examples: int = 3,
    ):
        self.device = device
        self.data_root = data_root
        self.output_dir = output_dir
        self.k_examples = k_examples
        os.makedirs(output_dir, exist_ok=True)

        print("Loading models...")
        # Initialize Grounding DINO
        self.gdino_model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth").to(device)
        self.gdino_model.eval()
        self.feature_extractor = GDINOFeatureExtractor(self.gdino_model, device)

        # Initialize LLaVA
        self.processor = LlavaNextProcessor.from_pretrained(llava_model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            llava_model_id, torch_dtype=torch.float16, device_map=device
        )

        # Load retrievers
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
                embedding_dim=20,
                device=self.device,
                mode="grounding",
            )
            retriever.load_index(task_dir)
            retrievers[task_type] = retriever

        return retrievers

    def get_prompt(self, task_type: str, retrieved_examples: List[Dict] = None) -> str:
        base_prompts = {
            "general_perception": (
                "A chat between a curious human and an autonomous driving expert. "
                "Focus on detected objects and their significance for the ego car. USER: <image>\n"
                "Please describe each object's type, position, and its impact. EXPERT:"
            ),
            "region_perception": (
                "A chat between a curious human and an autonomous driving expert. USER: <image>\n"
                "Please describe the objects detected in the given region and explain their importance. EXPERT:"
            ),
            "driving_suggestion": (
                "A chat between a curious human and an autonomous driving expert. USER: <image>\n"
                "Please provide driving suggestions based on detected objects. EXPERT:"
            ),
        }

        if not retrieved_examples:
            return base_prompts[task_type]

        prompt = "Here are some similar examples:\n\n"
        for i, example in enumerate(retrieved_examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Detected Objects: {example['classes']}\n"
            prompt += f"Bounding Boxes: {example['bounding_boxes']}\n\n"

        prompt += "\nNow, based on these examples, " + base_prompts[task_type]
        return prompt

    def process_single_example(self, example: Dict, task_type: str):
        try:
            # Fix image path by using basename
            image_name = os.path.basename(example["image"])
            image_path = os.path.join(self.data_root, "images", image_name)
            image = Image.open(image_path)

            # Preprocess image
            image_tensor = load_image(image_path).to(self.device)

            # Extract features using Grounding DINO
            features = self.feature_extractor.extract_features(image_tensor)

            # Retrieve similar examples
            retrieved_examples, _ = self.retrievers[task_type].retrieve(
                features["embedding_for_faiss"], 
                k=self.k_examples
            )

            # Generate prompt with retrieved examples
            prompt = self.get_prompt(task_type, retrieved_examples)

            # Prepare inputs for LLaVA
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
                self.device
            )

            # Generate response
            with torch.cuda.amp.autocast():
                output = self.model.generate(
                    **inputs, max_new_tokens=500, do_sample=False, use_cache=True
                )

            result = self.processor.decode(output[0], skip_special_tokens=True)
            result = result.split("EXPERT:", 1)[1] if "EXPERT:" in result else result

            return {"question_id": example["id"], "answer": result.strip()}

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

                # Save intermediate results periodically
                if i % 5 == 0:
                    self.save_results(intermediate=True)

        # Save final results
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
    parser.add_argument(
        "--llava_model_id", type=str, default="llava-hf/llava-v1.6-vicuna-7b-hf"
    )
    parser.add_argument(
        "--gdino_model_id", type=str, default="path/to/grounding_dino_model"
    )
    parser.add_argument(
        "--retrieval_indexes_dir", type=str, default="retrieval_indexes"
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="rag_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k_examples", type=int, default=3)

    args = parser.parse_args()

    processor = GDINORAGInference(
        llava_model_id=args.llava_model_id,
        gdino_model_id=args.gdino_model_id,
        retrieval_indexes_dir=args.retrieval_indexes_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        k_examples=args.k_examples,
    )

    processor.process_all_tasks()


if __name__ == "__main__":
    main()
