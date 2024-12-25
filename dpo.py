# %reload_ext autoreload
# %autoreload 2
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from trl import DPOConfig, DPOTrainer

peft_config = LoraConfig(
    base_model_name_or_path="llava-hf/llava-1.5-7b-hf",
    r=2,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
    ],
    bias="none",
)


model = AutoModelForVision2Seq.from_pretrained(
    "fine_tuned_results/lora_epoch_0",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
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
)
model.gradient_checkpointing_enable()
model.config.use_cache = False

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

dataset = load_dataset("json", data_files="preference_dataset.json")


def transform_data(example):
    example["prompt"] = example["prompt"][0]["content"]
    example["chosen"] = example["chosen"][0]["content"]
    example["rejected"] = example["rejected"][0]["content"]
    return example


dataset = dataset.map(transform_data)

training_args = DPOConfig(
    output_dir="dpo_results",
    logging_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    fp16=True,
)

trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=processor,
    peft_config=peft_config,
)

trainer.train()
