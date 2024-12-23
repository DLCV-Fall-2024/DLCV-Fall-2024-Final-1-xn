import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import faiss
import numpy as np
import torch


class TaskRetriever:
    def __init__(
        self,
        task_type: str,
        embedding_dim: int,
        device: str = "cuda",
        mode: str = "depth",  # Added mode to specify retrieval type
        index_path: str = None,
        similarity_weights: Dict[str, float] = None,
    ):
        """
        Initialize task-specific retriever

        Args:
            task_type: One of ['general', 'region', 'driving']
            embedding_dim: Dimension of the feature embeddings
            device: Device to store embeddings
            mode: 'depth' (ordinary) or 'grounding' (top-k bboxes + mIoU rerank)
            index_path: Path to load/save faiss index
            similarity_weights: Weights for different feature types
        """
        self.task_type = task_type
        self.device = device
        self.embedding_dim = embedding_dim
        self.mode = mode

        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(
            embedding_dim
        )  # Inner product = cosine similarity for normalized vectors
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # Set default similarity weights based on task
        self.similarity_weights = similarity_weights or self._get_default_weights()

        # Storage for examples
        self.examples = []
        self.total_examples = 0

        # Load existing index if provided
        if index_path and Path(index_path).exists():
            self.load_index(index_path)

    def _get_default_weights(self) -> Dict[str, float]:
        """Get default feature weights based on task type"""
        if self.task_type == "region":
            return {
                "global_features": 0.3,
                "spatial_features": 0.5,
                "depth_features": 0.2,
            }
        elif self.task_type == "driving":
            return {
                "global_features": 0.4,
                "spatial_features": 0.3,
                "depth_features": 0.3,
            }
        else:  # general perception
            return {
                "global_features": 0.4,
                "spatial_features": 0.4,
                "depth_features": 0.2,
            }

    def add_example(self, features: torch.Tensor, example_data: Dict):
        """
        Add a new example to the index

        Args:
            features: Normalized feature tensor
            example_data: Dictionary containing example information
        """
        # Convert to numpy and normalize
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        features = features.reshape(1, -1)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        # Add to FAISS index
        self.index.add(features)

        # Store example data
        self.examples.append(example_data)
        self.total_examples += 1

    def add_batch(self, features_batch: torch.Tensor, examples_batch: List[Dict]):
        """Add a batch of examples to the index"""
        if isinstance(features_batch, torch.Tensor):
            features_batch = features_batch.cpu().numpy()

        features_batch = features_batch / np.linalg.norm(
            features_batch, axis=1, keepdims=True
        )

        self.index.add(features_batch)
        self.examples.extend(examples_batch)
        self.total_examples += len(examples_batch)

    def retrieve(
        self, query_features: torch.Tensor, k: int = 5, query_bboxes: List[List[float]] = None
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Retrieve k most similar examples

        Args:
            query_features: Query feature tensor
            k: Number of examples to retrieve
            query_bboxes: List of bounding boxes for grounding mode

        Returns:
            Tuple of (retrieved examples, similarity scores)
        """
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.cpu().numpy()

        query_features = query_features.reshape(1, -1)
        query_features = query_features / np.linalg.norm(
            query_features, axis=1, keepdims=True
        )

        # Search FAISS index
        scores, indices = self.index.search(query_features, min(k, self.total_examples))

        # # Get corresponding examples
        # retrieved_examples = [self.examples[idx] for idx in indices[0]]

        # return retrieved_examples, scores[0]
        if self.mode == "depth":
            return self._retrieve_depth(indices, scores, k)
        elif self.mode == "grounding":
            if query_bboxes is None:
                query_bboxes = []
            return self._retrieve_grounding(query_features, query_bboxes, indices, scores, k)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def save_index(self, save_dir: str):
        """Save FAISS index and examples to disk"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if torch.cuda.is_available():
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(
                cpu_index, str(save_dir / f"{self.task_type}_index.faiss")
            )
        else:
            faiss.write_index(
                self.index, str(save_dir / f"{self.task_type}_index.faiss")
            )

        # Save examples and metadata
        with open(save_dir / f"{self.task_type}_examples.pkl", "wb") as f:
            pickle.dump(
                {
                    "examples": self.examples,
                    "total_examples": self.total_examples,
                    "similarity_weights": self.similarity_weights,
                },
                f,
            )

    def load_index(self, load_dir: str):
        """Load FAISS index and examples from disk"""
        load_dir = Path(load_dir)

        # Load FAISS index
        cpu_index = faiss.read_index(str(load_dir / f"{self.task_type}_index.faiss"))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

        # Load examples and metadata
        with open(load_dir / f"{self.task_type}_examples.pkl", "rb") as f:
            data = pickle.load(f)
            self.examples = data["examples"]
            self.total_examples = data["total_examples"]
            self.similarity_weights = data["similarity_weights"]

    def _retrieve_depth(
        self, indices: np.ndarray, scores: np.ndarray, k: int
    ) -> Tuple[List[Dict], np.ndarray]:
        retrieved_examples = [self.examples[idx] for idx in indices[0]]
        return retrieved_examples, scores[0]

    #add for ground dino
    def _retrieve_grounding(
        self,
        query_features: np.ndarray,
        query_bboxes: List[List[float]],
        indices: np.ndarray,
        scores: np.ndarray,
        k: int
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        rerank (mIoU)
        """
        initial_results = []
        for idx, score in zip(indices[0], scores[0]):
            example = self.examples[idx]
            # Save initial score
            initial_results.append((score, example))

        # new_score = 0.7 * score + 0.3 * mean_iou
        reranked = []
        for score, example in initial_results:
            db_bboxes = example.get("bounding_boxes", [])
            iou_score = self._mean_iou(query_bboxes, db_bboxes)
            # Combine scores
            new_score = 0.7 * score + 0.3 * iou_score
            reranked.append((new_score, example))

        # Sort by new score
        reranked.sort(key=lambda x: x[0], reverse=True)

        top_k = reranked[:k]
        top_examples = [r[1] for r in top_k]
        top_scores = [r[0] for r in top_k]
        return top_examples, top_scores

    def _mean_iou(self, bboxes1: List[List[float]], bboxes2: List[List[float]]) -> float:
        """
        Calculate mIoU of two classes of bounding boxes
        bboxes1, bboxes2: list of [x1, y1, x2, y2]
        """
        if not bboxes1 or not bboxes2:
            return 0.0

        total_iou = 0.0
        count = 0
        for boxA in bboxes1:
            for boxB in bboxes2:
                total_iou += self._iou(boxA, boxB)
                count += 1

        return total_iou / count if count > 0 else 0.0

    def _iou(self, boxA, boxB) -> float:
        """
        Calculate IoU of two bounding boxes
        boxA, boxB: [x1, y1, x2, y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = boxAArea + boxBArea - interArea

        return interArea / union if union > 0 else 0.0