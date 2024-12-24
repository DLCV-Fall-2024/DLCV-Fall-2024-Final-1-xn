from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from trl import DPOConfig, DPOTrainer

model = AutoModelForVision2Seq.from_pretrained("fine_tuned_results/lora_epoch_1")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

dataset = load_dataset("json", data_files="data.json")

training_args = DPOConfig(output_dir="dpo_results", logging_steps=10)

trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
)

trainer.train()
