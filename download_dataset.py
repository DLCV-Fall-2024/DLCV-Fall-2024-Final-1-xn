# download_dataset.py
import json
import os

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_dataset():
    # Create data directories
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/annotations", exist_ok=True)

    # Download dataset for each split with correct split names
    splits = ["train", "val", "test"]  # Changed from "validation" to "val"

    for split in splits:
        print(f"\nDownloading {split} split...")
        try:
            # Load dataset
            dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split)

            # Prepare annotations
            general_data = []
            region_data = []
            driving_data = []

            # Process each example
            for item in tqdm(dataset):
                # Save image
                image_filename = f"{item['id']}.jpg"
                image_path = os.path.join("data/images", image_filename)
                item["image"].save(image_path)

                # Create annotation
                annotation = {
                    "id": item["id"],
                    "image": image_path,
                    "conversations": item["conversations"],
                }

                # Sort into appropriate task
                if "general" in item["id"].lower():
                    general_data.append(annotation)
                elif "region" in item["id"].lower():
                    region_data.append(annotation)
                elif "suggestion" in item["id"].lower():
                    driving_data.append(annotation)

            # Save annotations with correct split name
            annotation_split = split
            for task, data in [
                ("general_perception", general_data),
                ("region_perception", region_data),
                ("driving_suggestion", driving_data),
            ]:
                output_file = os.path.join(
                    "data/annotations", f"{annotation_split}_{task}.jsonl"
                )
                with open(output_file, "w") as f:
                    for item in data:
                        json.dump(item, f)
                        f.write("\n")

            print(f"Completed {split} split")

            # Print some statistics
            print(f"Number of examples:")
            print(f"  General Perception: {len(general_data)}")
            print(f"  Region Perception: {len(region_data)}")
            print(f"  Driving Suggestion: {len(driving_data)}")

        except Exception as e:
            print(f"Error processing {split} split: {str(e)}")
            continue


if __name__ == "__main__":
    download_dataset()
