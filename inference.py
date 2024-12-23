import json
import os
import shutil

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Configuration
MAX_TOKEN = 300
OUTPUT_DIR = "inference_results"
FINE_TUNED_MODEL_DIR = "fine_tuned_results/lora_epoch_1"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Set higher memory fraction for RTX 4090
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cuda.matmul.allow_tf32 = True


def create_submission_zip():
    api_key_path = os.path.join(OUTPUT_DIR, "api_key.txt")
    with open(api_key_path, "w") as f:
        f.write("YOUR_GEMINI_API_KEY")

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copy2(os.path.join(OUTPUT_DIR, "submission.json"), temp_dir)
        shutil.copy2(api_key_path, temp_dir)

        # Create zip from temporary directory
        zip_path = os.path.join(OUTPUT_DIR, "pred")
        shutil.make_archive(zip_path, "zip", temp_dir)

    print(f"Submission zip created at {zip_path}.zip")


class DataProcessor:
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
        print("Setting up model...")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(
                f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB"
            )

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )

        self.model = PeftModel.from_pretrained(
            self.model, FINE_TUNED_MODEL_DIR, torch_dtype=torch.float16
        )
        self.model.eval()
        self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        print("Model loaded successfully with fine-tuned LoRA weights")

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
            "regional": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "recognizing traffic scenes and making detailed explanation. The expert receives an "
                "image of traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please describe the object inside the red rectangle in the image and explain why it "
                "affect ego car driving. EXPERT:"
            ),
            "suggestion": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "providing specific and helpful driving suggestions. The expert receives an image of "
                "traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please provide driving suggestions for the ego car based on the current scene. EXPERT:"
            ),
        }
        if task_type == "general":
            return prompts["general"]
        elif task_type == "regional":
            return prompts["regional"]
        elif task_type == "suggestion":
            return prompts["suggestion"]
        else:
            raise ValueError(f"Invalid task type: {task_type}")

    def process_single_example(self, example):
        images = example["image"]
        prompt = self.get_prompt(example["id"].split("_")[1])

        inputs = self.processor(
            text=prompt, images=images, return_tensors="pt", padding=True
        )

        with torch.no_grad(), torch.amp.autocast("cuda"):
            output = self.model.generate(
                **inputs, max_new_tokens=MAX_TOKEN, do_sample=False, use_cache=True
            )

        result = self.processor.decode(output[0], skip_special_tokens=True)
        result = result.split("EXPERT: ", 1)[1] if "EXPERT: " in result else result

        return {"id": example["id"], "answer": result}

    def process_all_tasks(self):
        dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming=True)

        for i, example in enumerate(tqdm(dataset)):
            result = self.process_single_example(example)
            if result:
                self.all_results[result["id"]] = result["answer"]

                if i % 5 == 0:
                    self.save_results(intermediate=True)

        self.save_results(intermediate=False)
        create_submission_zip()


def main():
    processor = DataProcessor()
    processor.process_all_tasks()
    print("\nInference complete!")


if __name__ == "__main__":
    main()
