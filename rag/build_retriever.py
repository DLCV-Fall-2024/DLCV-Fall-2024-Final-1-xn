import argparse
import json
import os
from pathlib import Path

import torch
from feature_extractor import FeatureExtractor
from PIL import Image
from task_retriever import TaskRetriever
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from depth_anything.dpt import DepthAnything


class CODALMDataset(Dataset):
    """Dataset class that works with the downloaded CODA-LM data format"""

    def __init__(self, data_root: str, split: str, task_type: str):
        """
        Args:
            data_root: Root directory containing images and annotations folders
            split: 'train', 'val', or 'test'
            task_type: 'general_perception', 'region_perception', or 'driving_suggestion'
        """
        self.data_root = Path(data_root)
        self.split = split
        self.task_type = task_type

        # Load annotations
        anno_path = self.data_root / "annotations" / f"{split}_{task_type}.jsonl"
        self.data = []
        with open(anno_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = str(self.data_root / "images" / os.path.basename(item["image"]))
        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "id": item["id"],
            "answer": item["conversations"][1]["value"]
            if "conversations" in item
            else item.get("answer", ""),
            "question": item["conversations"][0]["value"]
            if "conversations" in item
            else item.get("question", ""),
        }


def collate_fn(batch):
    """
    Custom collate function to handle PIL images and other data types
    """
    images = []
    ids = []
    answers = []
    questions = []

    for item in batch:
        images.append(item["image"])
        ids.append(item["id"])
        answers.append(item["answer"])
        questions.append(item["question"])

    return {
        "image": images,  # Keep as list of PIL images
        "id": ids,
        "answer": answers,
        "question": questions,
    }


def prepare_retrievers(
    data_root: str,
    save_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = "cuda",
    max_examples: int = None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing models...")
    depth_model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").to(
        device
    )
    depth_model.eval()
    feature_extractor = FeatureExtractor(depth_model, device)

    task_types = ["general_perception", "region_perception", "driving_suggestion"]

    for task_type in task_types:
        print(f"\nProcessing {task_type}")

        dataset = CODALMDataset(data_root, "train", task_type)
        if max_examples:
            dataset.data = dataset.data[:max_examples]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn,  # Add custom collate function
        )

        embedding_dim = feature_extractor.get_embedding_dim()
        retriever = TaskRetriever(
            task_type=task_type, embedding_dim=embedding_dim, device=device
        )

        for batch in tqdm(dataloader, desc=f"Building {task_type} index"):
            # Process list of PIL images
            images = [feature_extractor.preprocess_image(img) for img in batch["image"]]
            image_batch = torch.stack(images).to(device)

            # Extract features
            with torch.no_grad():
                features = feature_extractor.extract_and_concatenate(image_batch)

            # Prepare example data
            examples = [
                {"id": id_, "answer": answer, "question": question}
                for id_, answer, question in zip(
                    batch["id"], batch["answer"], batch["question"]
                )
            ]

            retriever.add_batch(features, examples)

        print(f"Saving {task_type} index...")
        retriever.save_index(save_dir / task_type)
        print(f"Processed {retriever.total_examples} examples for {task_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Build retrieval indexes for downloaded CODA-LM dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing images and annotations folders",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save indexes"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum examples per task (for testing)",
    )

    args = parser.parse_args()

    prepare_retrievers(
        data_root=args.data_root,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
