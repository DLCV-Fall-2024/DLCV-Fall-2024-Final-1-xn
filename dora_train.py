import argparse
import os
import re

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from liger_kernel.transformers import apply_liger_kernel_to_llama
from peft import LoraConfig, get_peft_model
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

apply_liger_kernel_to_llama()


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_prompt(task_type):
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
        "regional": (
            "A chat between a curious human and an autonomous driving expert, specializing in "
            "recognizing traffic scenes and making detailed explanations. The expert receives an "
            "image of traffic captured from the perspective of the ego car. USER: <image>\n"
            "Please describe the object inside the red rectangle in the image and explain why it "
            "affects ego car driving. EXPERT:"
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
        print(f"Invalid task type: {task_type}")
        raise ValueError


def format_conversations(examples):
    return [
        f"{get_prompt(examples[i]['id'].split('_')[1])} {examples[i]['conversations'][1]['value']}"
        for i in range(len(examples))
    ]


def train_collate_fn(examples):
    images = [example["image"] for example in examples]
    texts = format_conversations(examples)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["labels"],
    )


def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []

    for example in examples:
        image = example["image"]
        ground_truth = example["conversations"][1]["value"]
        images.append(image)
        prompt = get_prompt(example["id"].split("_")[1])
        texts.append(prompt)
        answers.append(ground_truth)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers


class LlavaTransformer(pl.LightningModule):
    def __init__(self, model, processor, learning_rate, max_length):
        super().__init__()
        self.processor = processor
        self.model = model

        self.learning_rate = learning_rate
        self.max_length = max_length

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = outputs.loss

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, answers = batch
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=self.max_length,
        )
        predictions = self.processor.batch_decode(
            generated_ids[:, input_ids.size(1) :], skip_special_tokens=True
        )

        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?<=>) | (?=</s_)", "", pred)

            print(f"Prediction: {pred}")
            print(f"    Answer: {answer}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        save_dir = os.path.join(output_dir, f"lora_epoch_{self.current_epoch}")
        self.model.save_pretrained(save_dir)
        print(f"LoRA model saved at {save_dir}")


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
parser.add_argument("--data_path", type=str, default="ntudlcv/dlcv_2024_final1")
parser.add_argument("--output_dir", type=str, default="fine_tuned_results")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument(
    "--num_epochs", type=int, default=2, help="Number of training epochs"
)
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument(
    "--max_length", type=int, default=300, help="Cutoff length for tokenization"
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to use for training"
)
parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
parser.add_argument(
    "--lora_dropout", type=float, default=0.05, help="LoRA dropout rate"
)

args = parser.parse_args()

model_id = args.model_id
data_path = args.data_path
output_dir = args.output_dir
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
max_length = args.max_length
device = args.device
lora_r = args.lora_r
lora_alpha = args.lora_alpha
lora_dropout = args.lora_dropout

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_token = os.getenv("HF_TOKEN")

device = torch.device(device)
print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained(model_id, torch_dtype=torch.float16)
processor.tokenizer.padding_side = "right"

pretrained_model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    token=hf_token,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    _attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
)

target_modules_list = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

lora_config = LoraConfig(
    use_dora=True,
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=find_all_linear_names(pretrained_model),
    # target_modules=target_modules_list,
    lora_dropout=lora_dropout,
    bias="none",
)

pretrained_model = get_peft_model(pretrained_model, lora_config)
pretrained_model.to(device)

dataset = load_dataset(data_path)
del dataset["test"]

train_dataloader = DataLoader(
    dataset["train"],
    batch_size=batch_size,
    collate_fn=train_collate_fn,
    shuffle=True,
)

val_dataloader = DataLoader(
    dataset["val"],
    batch_size=batch_size,
    collate_fn=eval_collate_fn,
)

model = LlavaTransformer(pretrained_model, processor, learning_rate, max_length)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=num_epochs,
    precision="16-mixed",
    limit_val_batches=4,
    accumulate_grad_batches=4,
    callbacks=[ModelCheckpoint(save_top_k=-1, save_on_train_epoch_end=True)],
)

trainer.fit(model, train_dataloader, val_dataloader)
