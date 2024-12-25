import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForVision2Seq, AutoProcessor

from trl import DPOConfig, DPOTrainer

peft_config = LoraConfig(
    base_model_name_or_path="llava-hf/llava-1.5-7b-hf",
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",  # Specify bias option ("none", "all", or "lora_only")
    task_type="SEQ_2_SEQ_LM",  # Task type: Adjust to match your use case
)


model = AutoModelForVision2Seq.from_pretrained(
    "fine_tuned_results/lora_epoch_0", torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

dataset = load_dataset("json", data_files="preference_dataset.json")

training_args = DPOConfig(
    output_dir="dpo_results",
    logging_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
)

trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=processor,
    peft_config=peft_config,
)

trainer.train()
