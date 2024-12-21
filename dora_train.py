import os

import torch
from datasets import load_dataset, load_from_disk
from liger_kernel.transformers import apply_liger_kernel_to_llama
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    LlavaNextForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)

apply_liger_kernel_to_llama()


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
        ),
    }
    return prompts.get(task_type, "")


def train_model(
    base_model: str,
    data_path: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: int,
    use_dora: bool,
    quantize: bool,
    eval_step: int,
    save_step: int,
    device: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    hub_model_id: str,
    push_to_hub: bool,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")

    # Setup device
    device = torch.device(device)
    print(f"Using device: {device}")

    # load tokenizer
    processor = LlavaProcessor.from_pretrained(base_model, token=hf_token)

    # QDoRA (quantized dora): IF YOU WANNA QUANTIZE THE MODEL
    if quantize:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model,
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
        )
        # setup for quantized training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model, token=hf_token
        )

    if lora_target_modules:
        target_modules_list = lora_target_modules.split(",")
    else:
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
        use_dora=use_dora,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules_list,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # get the peft model with LoRa config
    model = get_peft_model(model, lora_config)

    model.to(device)  # MODEL TO GPU/CUDA

    # Load the dataset
    dataset = load_dataset(data_path)
    # test set will cause error during mapping because it has no labels
    del dataset["test"]

    def format_conversations(examples):
        return [
            f"Human: <image>\n{get_prompt(examples['id'][i].split('_')[1])}\nAssistant: {examples['conversations'][i][1]['value']}"
            for i in range(len(examples["id"]))
        ]

    def tokenize_function(examples):
        text_input = format_conversations(examples)
        inputs = processor(
            text=text_input,
            images=examples["image"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    # Tokenize the dataset and prepare for training
    if os.path.exists("tokenized_data"):
        tokenized_datasets = load_from_disk("tokenized_data")
    else:
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        tokenized_datasets.save_to_disk("tokenized_data")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor,
        mlm=False,
        return_tensors="pt",
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        gradient_accumulation_steps=16,
        fp16=True,
        learning_rate=learning_rate,
        hub_token=hf_token,
    )

    # Clear CUDA cache to free memory
    torch.cuda.empty_cache()

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
    )

    # Start model training
    trainer.train()

    # Save and push the trained model and tokenizer
    if push_to_hub:
        # Push the main model to the hub
        trainer.push_to_hub(commit_message="Fine-tuned model")

    # Save the model and tokenizer locally
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_path", type=str, default="ntudlcv/dlcv_2024_final1")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_llava")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--cutoff_len", type=int, default=300, help="Cutoff length for tokenization"
    )
    parser.add_argument(
        "--val_set_size", type=int, default=8716, help="Validation set size"
    )
    parser.add_argument("--use_dora", action="store_true", help="Apply Dora")
    parser.add_argument("--quantize", action="store_true", help="Use quantization")
    parser.add_argument(
        "--eval_step", type=int, default=10, help="Evaluation step interval"
    )
    parser.add_argument("--save_step", type=int, default=100, help="Save step interval")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout rate"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated list of target modules for LoRA",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="path/to/repo",
        help="Repository name to push the model on the Hugging Face Hub",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to Hugging Face Hub",
    )

    args = parser.parse_args()
    train_model(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        use_dora=args.use_dora,
        quantize=args.quantize,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
    )
