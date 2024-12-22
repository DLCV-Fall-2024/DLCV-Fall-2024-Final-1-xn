import os
import json
from datasets import load_dataset
from tqdm import tqdm

def format_conversation(prompt, response_text):
    """Format a single conversation pair into LLaVA format."""
    return [
        {
            "from": "human",
            "value": prompt
        },
        {
            "from": "gpt",
            "value": response_text
        }
    ]

def download_and_format_dataset():
    # Create directories
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/annotations", exist_ok=True)
    
    # Define base prompts
    base_prompts = {
        "general_perception": (
            "A chat between a curious human and an autonomous driving expert, specializing in "
            "recognizing traffic scenes and making detailed explanation. The expert receives an "
            "image of traffic captured from the perspective of the ego car. USER: <image>\n "
            "Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, "
            "buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), "
            "traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), "
            "traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not "
            "discuss any objects beyond the seven categories above. Please describe each object's "
            "color, position, status, implication, respones, and how they influence ego car. EXPERT:"
        ),
        "region_perception": (
            "A chat between a curious human and an autonomous driving expert, specializing in "
            "recognizing traffic scenes and making detailed explanation. The expert receives an "
            "image of traffic captured from the perspective of the ego car. USER: <image>\n"
            "Please describe the object inside the red rectangle in the image and explain why it "
            "affect ego car driving. EXPERT:"
        ),
        "driving_suggestion": (
            "A chat between a curious human and an autonomous driving expert, specializing in "
            "providing specific and helpful driving suggestions. The expert receives an image of "
            "traffic captured from the perspective of the ego car. USER: <image>\n"
            "Please provide driving suggestions for the ego car based on the current scene. EXPERT:"
        )
    }

    # Process each split
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        try:
            # Load dataset
            dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split)
            
            # Initialize data containers for each task
            formatted_data = {
                "general_perception": [],
                "region_perception": [],
                "driving_suggestion": []
            }
            
            # Process each example
            for item in tqdm(dataset):
                # Save image
                image_filename = f"{item['id']}.jpg"
                image_path = os.path.join("data/images", image_filename)
                item['image'].save(image_path)
                
                # Determine task type from ID
                task_type = None
                if "general" in item['id'].lower():
                    task_type = "general_perception"
                elif "region" in item['id'].lower():
                    task_type = "region_perception"
                elif "suggestion" in item['id'].lower():
                    task_type = "driving_suggestion"
                
                if task_type:
                    # Get the response text from conversations
                    response_text = item['conversations']  # The dataset provides the response directly
                    
                    # Format in LLaVA style
                    formatted_entry = {
                        "id": item['id'],
                        "image": image_filename,  # Relative path as required by LLaVA
                        "conversations": format_conversation(
                            base_prompts[task_type],
                            response_text
                        )
                    }
                    formatted_data[task_type].append(formatted_entry)
            
            # Save formatted data for each task
            for task, data in formatted_data.items():
                output_file = os.path.join("data/annotations", f"{split}_{task}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"{task} - {split}: {len(data)} examples")
        
        except Exception as e:
            import traceback
            print(f"Error processing {split} split: {str(e)}")
            print(traceback.format_exc())  # Print the full error traceback
            continue

def merge_task_files(split, tasks, output_file):
    """Merge multiple task files into a single file for training."""
    merged_data = []
    
    for task in tasks:
        input_file = f"data/annotations/{split}_{task}.json"
        if os.path.exists(input_file):
            with open(input_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
                merged_data.extend(task_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created merged file {output_file} with {len(merged_data)} examples")

if __name__ == "__main__":
    # Download and format the dataset
    download_and_format_dataset()
    
    # Merge all tasks into single files for each split
    tasks = ["general_perception", "region_perception", "driving_suggestion"]
    
    for split in ["train", "val", "test"]:
        merge_task_files(
            split=split,
            tasks=tasks,
            output_file=f"data/annotations/{split}_merged.json"
        )