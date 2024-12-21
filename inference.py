import gc
import json
import os
import time

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

# Configuration
MAX_TOKEN = 300
OUTPUT_DIR = "inference_results"
FINE_TUNED_MODEL_DIR = "fine_tuned_results/lora_final"
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
DATA_ROOT = "data"
DEBUG = True

# Set higher memory fraction for RTX 4090
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cuda.matmul.allow_tf32 = True


class LocalDataProcessor:
    def __init__(self):
        self.setup_model()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.all_results = self.load_existing_results()

    def load_existing_results(self):
        """Load existing results from submission.json if it exists"""
        submission_path = os.path.join(OUTPUT_DIR, "submission.json")
        if os.path.exists(submission_path):
            with open(submission_path, "r") as f:
                return json.load(f)
        return {}

    def save_results(self, intermediate=True):
        """Save current results to file"""
        if intermediate:
            save_path = os.path.join(OUTPUT_DIR, "submission_intermediate.json")
        else:
            save_path = os.path.join(OUTPUT_DIR, "submission.json")

        with open(save_path, "w") as f:
            json.dump(self.all_results, f, indent=2)

        if not intermediate:
            print(f"Final results saved to {save_path}")

    def setup_model(self):
        """Initialize model optimized for RTX 4090"""
        print("Setting up model...")
        try:
            torch.cuda.empty_cache()
            gc.collect()

            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                print(
                    f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB"
                )

            self.processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                low_cpu_mem_usage=True,
            )

            # Load the LoRA adapter
            if not os.path.exists(FINE_TUNED_MODEL_DIR):
                raise FileNotFoundError(
                    f"Fine-tuned model weights not found at {FINE_TUNED_MODEL_DIR}"
                )

            self.model = PeftModel.from_pretrained(
                self.model, FINE_TUNED_MODEL_DIR, torch_dtype=torch.float16
            )
            self.model.eval()
            self.model.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )

            self.device = torch.device("cuda:0")
            print(f"Using device: {self.device}")
            print("Model loaded successfully with fine-tuned LoRA weights")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_prompt(self, task_type):
        """Get appropriate prompt for task type"""
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

    def process_single_example(self, example, task_type):
        """Process a single example"""
        # Skip if already processed
        if example["id"] in self.all_results:
            print(f"Skipping already processed example {example['id']}")
            return {
                "question_id": example["id"],
                "answer": self.all_results[example["id"]],
            }

        try:
            image = Image.open(example["image"])
            prompt = self.get_prompt(task_type)

            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt", padding=True
            )

            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.no_grad(), torch.amp.autocast(
                "cuda"
            ):  # Fixed deprecated autocast
                output = self.model.generate(
                    **inputs, max_new_tokens=MAX_TOKEN, do_sample=False, use_cache=True
                )

            result = self.processor.decode(output[0], skip_special_tokens=True)
            result = result.split("EXPERT: ", 1)[1] if "EXPERT: " in result else result

            del inputs, output
            torch.cuda.empty_cache()
            gc.collect()

            return {"question_id": example["id"], "answer": result}

        except Exception as e:
            print(f"Error processing example {example['id']}: {e}")
            return None

    def process_all_tasks(self):
        """Process all tasks with correct filenames"""
        # Task mapping to correct filenames
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

            # Process each example
            for i, example in enumerate(tqdm(examples)):
                result = self.process_single_example(example, task_type)
                if result:
                    self.all_results[result["question_id"]] = result["answer"]

                    # Save intermediate results every 5 examples
                    if i % 5 == 0:
                        self.save_results(intermediate=True)

            # Save results after completing task
            self.save_results(intermediate=True)
            print(f"Completed {task_type} task, processed {len(examples)} examples")

        # Save final results and create submission
        self.save_results(intermediate=False)
        self.create_submission_zip()

    def create_submission_zip(self):
        """Create final submission zip file"""
        import os
        import shutil

        # Create api_key.txt
        api_key_path = os.path.join(OUTPUT_DIR, "api_key.txt")
        with open(api_key_path, "w") as f:
            f.write("YOUR_GEMINI_API_KEY")  # Replace with actual API key

        # Create temporary directory for zip contents
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy files to temporary directory
            shutil.copy2(os.path.join(OUTPUT_DIR, "submission.json"), temp_dir)
            shutil.copy2(api_key_path, temp_dir)

            # Create zip from temporary directory
            zip_path = os.path.join(OUTPUT_DIR, "pred")
            shutil.make_archive(zip_path, "zip", temp_dir)

        print(f"Submission zip created at {zip_path}.zip")


def main():
    try:
        processor = LocalDataProcessor()
        processor.process_all_tasks()
        print("\nInference complete!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
