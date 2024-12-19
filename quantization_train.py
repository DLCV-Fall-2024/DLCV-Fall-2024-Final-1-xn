import torch
from torchvision import transforms
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, BitsAndBytesConfig
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
import traceback

import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

GRADIENT_ACCUMULATION_STEPS = 4
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# Configuration
MAX_TOKEN = 100
OUTPUT_DIR = "fine_tuned_results"
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
DATA_ROOT = "data"
DEBUG = False
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
SAVE_INTERVAL = 1000

# Set GPU memory settings
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cuda.matmul.allow_tf32 = True

class DrivingDataset(Dataset):
    def __init__(self, annotations_file, root_dir, task_type):
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
        try:
            sample = self.samples[idx]
            image_path = sample['image']
            if self.root_dir.endswith('data') or self.root_dir.endswith('data/'):
                image_path = os.path.join(os.path.dirname(self.root_dir), image_path)

            # Load and verify image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            conversations = sample['conversations']
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
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return None

    def get_prompt(self, task_type):
        prompts = {
            "general": (
                "A chat between a curious human and an autonomous driving expert, specializing in "
                "recognizing traffic scenes and making detailed explanations. The expert receives an "
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
                "recognizing traffic scenes and making detailed explanations. The expert receives an "
                "image of traffic captured from the perspective of the ego car. USER: <image>\n"
                "Please describe the object inside the red rectangle in the image and explain why it "
                "affects ego car driving. EXPERT:"
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
        print("Setting up model...")
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB")

            self.processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

            self.processor.patch_size = 14
            self.processor.vision_feature_select_strategy = "full"
            
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # self.model = LlavaNextForConditionalGeneration.from_pretrained(
            #     MODEL_ID,
            #     quantization_config=bnb_config,
            #     torch_dtype=torch.float16,
            #     device_map="auto",
            #     low_cpu_mem_usage=True
            # )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID)

            # Apply Liger Kernel
            apply_liger_kernel_to_llama()

            # LoRA Configuration
            lora_config = LoraConfig(
                r=8,  
                lora_alpha=32,
                target_modules=["q_proj"],  # Reduced target modules
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            print("Model loaded successfully")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def collate_fn(self, batch):
        try:
            valid_samples = [item for item in batch if item is not None]
            if not valid_samples:
                return None
                
            images = [item['image'] for item in valid_samples]
            prompts = [item['prompt'] for item in valid_samples]
            labels = [item['label'] for item in valid_samples]
            
            # Process images with explicit type casting
            vision_inputs = self.processor.image_processor(
                images,
                return_tensors="pt",
                do_resize=True,
                size={"height": 224, "width": 224}
            )
            
            if vision_inputs.pixel_values is None:
                return None
                
            # Ensure pixel values are float32 and in range [0, 1]
            pixel_values = vision_inputs.pixel_values.to(torch.float32)
            if pixel_values.max() > 1.0:
                pixel_values = pixel_values / 255.0
                
            # Process text with max length limit
            text_inputs = self.processor.tokenizer(
                prompts,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=512  # Explicit max length
            )
            
            # Create inputs with explicit types
            inputs = {
                "input_ids": text_inputs["input_ids"].to(torch.long),
                "attention_mask": text_inputs["attention_mask"].to(torch.long),
                "pixel_values": pixel_values,
                "image_sizes": torch.tensor([[224, 224]] * len(valid_samples), dtype=torch.long)
            }
            
            # Process labels with same max length
            label_tokens = self.processor.tokenizer(
                labels,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=512  # Same max length as inputs
            )
            
            # Create labels tensor with padding
            labels_tensor = label_tokens["input_ids"].clone().to(torch.long)
            labels_tensor[label_tokens["attention_mask"] == 0] = -100
            inputs["labels"] = labels_tensor
            
            # Debug info
            '''for key, tensor in inputs.items():
                print(f"Debug - {key} shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
                if torch.isnan(tensor).any():
                    print(f"Warning: NaN values found in {key}")
                if torch.isinf(tensor).any():
                    print(f"Warning: Inf values found in {key}")'''
            
            return inputs

        except Exception as e:
            print(f"Error in collate_fn: {str(e)}")
            traceback.print_exc()
            return None

    def train_model(self, data_root=DATA_ROOT):
        """Complete training loop with improved error handling and debugging"""
        print("Starting training process...")
        print(f"Using data root: {data_root}")

        # Dataset loading
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

        combined_train_dataset = ConcatDataset(train_datasets)
        print(f"Total training samples: {len(combined_train_dataset)}")

        train_loader = DataLoader(
            combined_train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=2,  # Reduced for stability
            pin_memory=False,  # Disabled to prevent CUDA issues
            collate_fn=self.collate_fn
        )

        # Load validation datasets
        val_datasets = []
        for task_type, filename in val_tasks:
            annotations_file = os.path.join(data_root, "annotations", filename)
            dataset = DrivingDataset(annotations_file, data_root, task_type)
            val_datasets.append(dataset)

        combined_val_dataset = ConcatDataset(val_datasets)
        print(f"Total validation samples: {len(combined_val_dataset)}")

        val_loader = DataLoader(
            combined_val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=2,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01,
            eps=1e-8
        )
        
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=WARMUP_STEPS, 
            num_training_steps=total_steps
        )

        self.model.train()
        torch.cuda.empty_cache()

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            epoch_train_loss = 0.0
            valid_steps = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for step, batch in enumerate(progress_bar):
                try:
                    if batch is None:
                        print(f"Skipping step {step}: batch is None")
                        continue

                    # Clear cache before processing batch
                    if step % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                    # Move batch to GPU with careful type handling
                    cuda_batch = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            try:
                                if k == "pixel_values":
                                    tensor = v.to(device=self.device, dtype=torch.float16, non_blocking=True)
                                elif k == "labels":
                                    # Ensure labels are valid indices
                                    tensor = v.clamp(min=-100, max=self.model.config.vocab_size-1)
                                    tensor = tensor.to(device=self.device, dtype=torch.long, non_blocking=True)
                                else:
                                    tensor = v.to(device=self.device, dtype=torch.long, non_blocking=True)
                                
                                # Verify tensor
                                if torch.isnan(tensor).any():
                                    print(f"Warning: NaN values in {k}")
                                    continue
                                if torch.isinf(tensor).any():
                                    print(f"Warning: Inf values in {k}")
                                    continue
                                    
                                cuda_batch[k] = tensor
                                
                            except Exception as e:
                                print(f"Error moving tensor {k} to GPU: {str(e)}")
                                raise

                    # Forward pass with gradient scaling
                    try:
                        outputs = self.model(**cuda_batch)
                        loss = outputs.loss
                        
                        # Check loss validity
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Invalid loss value: {loss.item()}")
                            continue
                            
                        # Scale loss for gradient accumulation
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
                        loss.backward()

                    except Exception as e:
                        print(f"Error in forward/backward pass: {str(e)}")
                        continue

                    # Gradient accumulation step
                    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    # Update metrics
                    epoch_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                    valid_steps += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": loss.item() * GRADIENT_ACCUMULATION_STEPS,
                        "lr": scheduler.get_last_lr()[0],
                        "valid_steps": valid_steps
                    })

                    # Save checkpoint
                    if (step + 1) % SAVE_INTERVAL == 0:
                        self.save_lora_weights(epoch + 1, step + 1)

                except Exception as e:
                    print(f"Error during training step {step}: {str(e)}")
                    traceback.print_exc()
                    continue

            # End of epoch processing
            if valid_steps > 0:
                avg_train_loss = epoch_train_loss / valid_steps
                print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
                
                # Validation
                try:
                    val_loss = self.evaluate_model(val_loader)
                    print(f"Validation Loss: {val_loss:.4f}")
                except Exception as e:
                    print(f"Error during validation: {str(e)}")
                
                # Save epoch checkpoint
                self.save_lora_weights(epoch + 1)
                
                # Clear cache between epochs
                torch.cuda.empty_cache()
                gc.collect()

        # Save final model
        try:
            self.save_lora_weights('final')
            print("Training complete!")
        except Exception as e:
            print(f"Error saving final weights: {str(e)}")

    def evaluate_model(self, val_loader):
        print("Starting evaluation...")
        self.model.eval()
        total_val_loss = 0.0
        valid_steps = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                try:
                    if batch is None:
                        continue

                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    valid_steps += 1

                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    continue

        self.model.train()
        return total_val_loss / valid_steps if valid_steps > 0 else float('inf')

    def save_lora_weights(self, epoch, step=None):
        try:
            if step:
                save_path = os.path.join(OUTPUT_DIR, f"lora_epoch{epoch}_step{step}")
            else:
                save_path = os.path.join(OUTPUT_DIR, f"lora_{epoch}")
            self.model.save_pretrained(save_path)
            print(f"LoRA weights saved to {save_path}")
        except Exception as e:
            print(f"Error saving LoRA weights: {e}")

def main():
    try:
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