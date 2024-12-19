import torch
from torchvision import transforms
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
import gc
import os
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import time
import argparse

# Configuration
MAX_TOKEN = 1000
OUTPUT_DIR = "fine_tuned_results"
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
DATA_ROOT = "data"
DEBUG = False  # Set to True for debugging
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
SAVE_INTERVAL = 100  # Save every 100 steps

# Set higher memory fraction for RTX 4090
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cuda.matmul.allow_tf32 = True

class DrivingDataset(Dataset):
    def __init__(self, annotations_file, root_dir, task_type):
        """
        Initializes the dataset by loading the annotations.
        """
        self.root_dir = root_dir
        self.task_type = task_type
        self.samples = []
        
        try:
            with open(annotations_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.samples.append(data)
        except FileNotFoundError:
            print(f"Error: Could not find file {annotations_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the image and the corresponding ground truth for training.
        """
        sample = self.samples[idx]
        image_path = sample['image']
        # Modify for using Kaggle dataset
        # if the root_dir is data or data/, then the image_path is the image path
        if self.root_dir.endswith('data') or self.root_dir.endswith('data/'):
            image_path = os.path.join(os.path.dirname(self.root_dir), image_path)

        image = Image.open(image_path).convert("RGB")
        
        conversations = sample['conversations']
        
        # Extract the GPT response as label
        label = ""
        for convo in conversations:
            if convo['from'].lower() == 'gpt':
                label = convo['value']
                break

        prompt = self.get_prompt(self.task_type)
        
        return {
            "image": image,
            "prompt": prompt,
            "label": label
        }

    def get_prompt(self, task_type):
        """Get appropriate prompt for task type"""
        prompts = {
            "general": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "recognizing traffic scenes and providing detailed explanations. The expert receives an "
                "image of traffic captured from the perspective of the ego car. USER: <image>\n "
                "Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, "
                "buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), "
                "traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), "
                "traffic cones, barriers, miscellaneous (debris, dustbin, animals, etc.). You must not "
                "discuss any objects beyond the seven categories above. Please describe each object's "
                "color, position, status, implication, responses, and how they influence the ego car. EXPERT:"
            ),
            "region": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "recognizing traffic scenes and providing detailed explanations. The expert receives an "
                "image of traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please describe the object inside the red rectangle in the image and explain why it "
                "affects the ego car's driving. EXPERT:"
            ),
            "driving": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "providing specific and helpful driving suggestions. The expert receives an image of "
                "traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please provide driving suggestions for the ego car based on the current scene. EXPERT:"
            )
        }
        return prompts.get(task_type, "")

class LocalDataProcessor:
    def __init__(self):
        self.setup_model()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def setup_model(self):
        """Initialize model optimized for RTX 4090"""
        print("Setting up model...")
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB")

            self.processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
            
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                low_cpu_mem_usage=True
            )

            # LoRA Configuration
            lora_config = LoraConfig(
                target_modules=["q_proj", "v_proj"],
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Apply LoRA to the model
            self.model = get_peft_model(self.model, lora_config)
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            print("Model loaded successfully")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def collate_fn(self, batch):
        """Custom collate function to handle batching."""
        try:
            # images = torch.stack([item['image'] for item in batch])
            images = [item['image'] for item in batch]
            prompts = [item['prompt'] for item in batch]
            labels = [item['label'] for item in batch]
            
            # Tokenize prompts with images
            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            # Tokenize labels
            label_tokens = self.processor.tokenizer(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN
            ).input_ids
            
            inputs['labels'] = label_tokens
            
            return inputs

        except Exception as e:
            print(f"Error in collate_fn: {e}")
            raise

    def train_model(self, data_root=DATA_ROOT):
        """Fine-tune the model using LoRA on the training dataset"""
        print("Starting training process...")
        print(f"Using data root: {data_root}")

        # Prepare datasets for all three tasks
        train_tasks = [
            ("general", "train_general_perception.jsonl"),
            ("region", "train_region_perception.jsonl"),
            ("driving", "train_driving_suggestion.jsonl")
        ]

        val_tasks = [
            ("general", "val_general_perception.jsonl"),
            ("region", "val_region_perception.jsonl"),
            ("driving", "val_driving_suggestion.jsonl")
        ]

        # Load training datasets
        train_datasets = []
        for task_type, filename in train_tasks:
            annotations_file = os.path.join(data_root, "annotations", filename)
            dataset = DrivingDataset(annotations_file, data_root, task_type)
            train_datasets.append(dataset)

        # Combine all training datasets
        combined_train_dataset = ConcatDataset(train_datasets)
        print(f"Total training samples: {len(combined_train_dataset)}")

        train_loader = DataLoader(
            combined_train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        # Load validation datasets
        val_datasets = []
        for task_type, filename in val_tasks:
            annotations_file = os.path.join(data_root, "annotations", filename)
            dataset = DrivingDataset(annotations_file, data_root, task_type)
            val_datasets.append(dataset)

        # Combine all validation datasets
        combined_val_dataset = ConcatDataset(val_datasets)
        print(f"Total validation samples: {len(combined_val_dataset)}")

        val_loader = DataLoader(
            combined_val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        # Optimizer and Scheduler
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=WARMUP_STEPS, 
                                                    num_training_steps=total_steps)

        self.model.train()
        self.model.to(self.device)

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            epoch_train_loss = 0.0
            progress_bar = tqdm(train_loader, desc="Training", leave=False)

            for step, batch in enumerate(progress_bar):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss.backward()

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    epoch_train_loss += loss.item()
                    progress_bar.set_postfix({"loss": loss.item()})

                    # Save intermediate LoRA weights
                    if (step + 1) % SAVE_INTERVAL == 0:
                        self.save_lora_weights(epoch + 1, step + 1)

                except Exception as e:
                    print(f"Error during training step {step}: {e}")
                    continue

            avg_train_loss = epoch_train_loss / len(train_loader)
            print(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_train_loss:.4f}")

            # Evaluate on validation set after each epoch
            avg_val_loss = self.evaluate_model(val_loader)
            print(f"Epoch {epoch + 1} completed. Average Validation Loss: {avg_val_loss:.4f}")

            # Save LoRA weights at the end of each epoch
            self.save_lora_weights(epoch + 1, 'end_of_epoch')

        # Save final LoRA weights
        self.save_lora_weights('final')

        print("Training complete!")

    def evaluate_model(self, val_loader):
        """Evaluate the model on the validation dataset"""
        print("Starting evaluation on validation set...")
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_val_loss += loss.item()

                except Exception as e:
                    print(f"Error during evaluation step: {e}")
                    continue

        avg_val_loss = total_val_loss / len(val_loader)
        self.model.train()  # Set back to training mode
        return avg_val_loss

    def save_lora_weights(self, epoch, step=None):
        """Save LoRA adapter weights"""
        try:
            if step:
                save_path = os.path.join(OUTPUT_DIR, f"lora_epoch{epoch}_step{step}.pt")
            else:
                save_path = os.path.join(OUTPUT_DIR, f"lora_{epoch}.pt")
            self.model.save_pretrained(save_path)
            print(f"LoRA weights saved to {save_path}")
        except Exception as e:
            print(f"Error saving LoRA weights: {e}")

def main():
    try:
        # Add argument parsing
        parser = argparse.ArgumentParser(description='Train the model with a specified data root directory')
        parser.add_argument('--data_root', type=str, default=DATA_ROOT, help='Path to the data root directory')
        args = parser.parse_args()

        processor = LocalDataProcessor()
        processor.train_model(data_root=args.data_root)
        print("\nTraining and Evaluation complete!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()