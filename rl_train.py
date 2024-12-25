import os

from datasets import load_dataset
from peft import PeftModel
from pycocoevalcap.bleu.bleu import Bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from trl import PPOConfig, PPOTrainer


def clean_response(text):
    if "EXPERT: " in text:
        text = text.split("EXPERT: ", 1)[1]

    last_period_idx = text.rfind(".")
    if last_period_idx == -1:
        return text.strip() + "."

    cleaned_text = text[: last_period_idx + 1].strip()
    return cleaned_text


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
        raise ValueError(f"Invalid task type: {task_type}")


def format_conversations(examples):
    return [
        f"{get_prompt(examples[i]['id'].split('_')[1])} {examples[i]['conversations'][1]['value']}"
        for i in range(len(examples))
    ]


def collate_fn(examples):
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


def compute_reward(label, prediction):
    references = {0: [label]}
    hypotheses = {0: prediction}

    scorer = Bleu(n=3)  # Compute up to BLEU-3

    # Compute BLEU scores
    scores, _ = scorer.compute_score(references, hypotheses)

    # Extract BLEU-3 score (index 2 because BLEU-1 is at index 0, BLEU-2 at 1, etc.)
    bleu_3_score = scores[2]

    print(f"Label: {label}")
    print(f"Prediction: {prediction}")
    print(f"BLEU-3 Reward: {bleu_3_score}")

    return bleu_3_score


model_id = "llava-hf/llava-1.5-7b-hf"
dataset_name = "ntudlcv/dlcv_2024_final1"
output_dir = "rl_results"
os.makedirs(output_dir, exist_ok=True)

ppo_config = PPOConfig(
    learning_rate=1.4e-5,
    mini_batch_size=1,
    batch_size=1,
    gradient_accumulation_steps=1,
    seed=42,
    output_dir=output_dir,
)

dataset = load_dataset(dataset_name, split="train[:10]")

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor.tokenizer.padding_side = "right"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, _attn_implementation="flash_attention_2"
)
model = PeftModel.from_pretrained(model, "fine_tuned_results/lora_epoch_1")

dataloader = DataLoader(
    dataset["train"], batch_size=1, shuffle=True, collate_fn=collate_fn
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    processing_class=processor,
    dataset=dataset["train"],
)

for i, batch in enumerate(tqdm(dataloader)):
    outputs = ppo_trainer.generate(
        **batch, max_new_tokens=300, do_sample=False, use_cache=True
    )
    rewards = []
    for i in range(len(batch)):
        prediction = clean_response(
            processor.decode(outputs[i], skip_special_tokens=True)
        )
        label = processor.decode(batch[i]["label"], skip_special_tokens=True)
        reward = compute_reward(label, prediction)
        rewards.append(reward)

    ppo_trainer.step(batch["input_ids"], outputs, rewards)

ppo_trainer.save_model(output_dir)
