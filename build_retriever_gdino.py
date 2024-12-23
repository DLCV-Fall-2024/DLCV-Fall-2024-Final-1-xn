import os
import torch
from pathlib import Path
import tqdm
from torch.utils.data import DataLoader, Dataset
from groundingdino.util.inference import load_model, load_image, predict
from build_retriever import CODALMDataset, collate_fn
from task_retriever import TaskRetriever

class GDINOFeatureExtractor:
    def __init__(self, grounding_dino_model, device="cuda", top_k=5):
        self.model = grounding_dino_model
        self.device = device
        self.model.eval()
        self.top_k = top_k

    def extract_features(self, image: torch.Tensor) -> dict:
        """
        Extract object detection features using Grounding DINO.

        Args:
            image: Preprocessed image tensor.

        Returns:
            Dictionary containing bounding box and class information.
        """

        TEXT_PROMPT = "car, truck, bus, pedestrian, cyclist, motorcycle, animal, traffic light, stop sign, road sign, lane marking"

        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        with torch.no_grad():
            boxes, logits, _ = predict(
                model=self.model,
                image=image,
                device=self.device,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

        top_k_boxes = boxes[:self.top_k]
        top_k_logits = logits[:self.top_k]

        embed_dim = 4 * self.top_k
        box_embed = torch.zeros(embed_dim, device=self.device)

        for i, box in enumerate(top_k_boxes):
            start = i * 4
            box_embed[start : start + 4] = box  # [x1, y1, x2, y2]

        return {
            "embedding_for_faiss": box_embed.unsqueeze(0),  # shape = [1, 4*top_k]
            "bboxes_list": top_k_boxes.tolist(),
            "classes_list": top_k_logits.tolist()
        }

def build_retriever(
    data_root: str,
    save_dir: str,
    model_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = "cuda",
    max_examples: int = None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model and feature extractor
        gdino_model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", model_path).to("cuda")
        feature_extractor = GDINOFeatureExtractor(gdino_model, "cuda")
    except Exception as e:
        raise RuntimeError(f"Failed to load Grounding DINO model or feature extractor: {e}")


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
            collate_fn=collate_fn
        )

        retriever = TaskRetriever(
            task_type=task_type, embedding_dim=20, device=device, mode="grounding"
        )

        for batch in tqdm.tqdm(dataloader, desc=f"Building {task_type} index"):
            try:
                images = batch["images"]
                examples_batch = []  # Store structured examples
                features_batch = []  # Store embeddings for FAISS

                for idx, image in enumerate(images):
                    features = feature_extractor.extract_features(image)
                    meta_example = {
                        "image_id": batch["image_ids"][idx],
                        "bounding_boxes": features["bboxes_list"],  # list of bboxes
                        "classes": features["classes_list"],        # list of classes/logits
                }
                examples_batch.append(meta_example)
                features_batch.append(features["embedding_for_faiss"].squeeze(0))
                retriever.add_batch(torch.stack(features_batch, dim=0), examples_batch)
                features_batch = []
                examples_batch = []
            except Exception as e:
                print(f"Error processing batch: {e}")

        # Save retriever index for each task type
        retriever.save_index(save_dir / f"{task_type}_retriever")

if __name__ == "__main__":
    build_retriever(
        data_root="data/grounding_dino/",
        model_path="path/to/grounding_dino_model",
        save_dir="retrievers/gdino/",
    )