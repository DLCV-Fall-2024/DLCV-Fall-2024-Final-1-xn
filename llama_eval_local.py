import os
import json
import argparse
import numpy as np
from collections import defaultdict
import warnings
import logging
import re
from tqdm import tqdm

# Suppress warnings and logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import transformers
from scorer import Scorer

def load_local_data(data_root, split):
    """Load local data from jsonl files"""
    annotations_dir = os.path.join(data_root, "annotations")
    data = []
    
    for task in ["general_perception", "region_perception", "driving_suggestion"]:
        filename = f"{split}_{task}.jsonl"
        filepath = os.path.join(annotations_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
    
    return data

def get_reference_text(data, is_test=False):
    """Safely extract reference text from data in different formats"""
    try:
        if is_test:
            return None
            
        if "conversations" in data and isinstance(data["conversations"], list):
            if len(data["conversations"]) > 1:
                return data["conversations"][1]["value"]
        
        if "answer" in data:
            return data["answer"]
        
        if "question" in data and "ground_truth" in data:
            return data["ground_truth"]
            
        return None
        
    except Exception as e:
        return None

def create_prompt(message, is_test=False):
    """Create evaluation prompt specifically for autonomous driving scene descriptions"""
    system_context = (
        "You are an autonomous driving evaluation expert. "
        "Rate predictions on:\n"
        "1. Completeness: Covers all visible relevant objects\n"
        "2. Accuracy: Correctly describes positions and relationships\n"
        "3. Relevance: Focus on driving-relevant details\n"
        "Each aspect is equally important for the final score."
    )
    
    if is_test or message["reference"] is None:
        return (
            f"{system_context}\n\n"
            "Evaluate this prediction:\n\n"
            f"{message['prediction']}\n\n"
            "First provide a brief analysis of the prediction's strengths and weaknesses. "
            "Then give a rating from 1-10 where:\n"
            "1-3: Missing critical information or major inaccuracies\n"
            "4-6: Covers basics but lacks important details\n"
            "7-8: Good coverage with minor omissions\n"
            "9-10: Excellent, comprehensive coverage\n\n"
            "End your response with 'Rating: [[X]]' where X is your score.\n"
            "Example: 'Good coverage of visible objects but could better explain their impact on driving. Rating: [[7]]'"
        )
    else:
        return (
            f"{system_context}\n\n"
            "Compare this reference and prediction:\n\n"
            f"Reference:\n{message['reference']}\n\n"
            f"Prediction:\n{message['prediction']}\n\n"
            "First briefly analyze how well the prediction matches the reference in terms of completeness, "
            "accuracy, and relevance to driving. Then rate from 1-10 where:\n"
            "1-3: Major discrepancies or missing critical information\n"
            "4-6: Partial match with significant differences\n"
            "7-8: Good match with minor differences\n"
            "9-10: Excellent match\n\n"
            "End your response with 'Rating: [[X]]' where X is your score.\n"
            "Example: 'Prediction captures main elements but misses some important details. Rating: [[7]]'"
        )

def extract_score(text):
    """Extract score from model output with improved parsing"""
    try:
        pattern = r"Rating:\s*\[\[(\d+)\]\]"
        matches = list(re.finditer(pattern, text))
        if matches:
            score = int(matches[-1].group(1))
            if 1 <= score <= 10:
                return score
    except Exception as e:
        pass
    return None

def process_batch(pipeline, batch_data, args, is_test=False):
    """Process a batch of samples with improved error handling"""
    prompts = []
    batch_info = []
    
    for sample_id, data, pred in batch_data:
        reference_text = get_reference_text(data, is_test)
        message = {
            "reference": reference_text,
            "prediction": pred
        }
        prompt = create_prompt(message, is_test)
        prompts.append(prompt)
        batch_info.append((sample_id, (sample_id.split("_")[1]).lower()))
    
    try:
        outputs = pipeline(
            prompts,
            max_new_tokens=args.max_output_tokens,
            do_sample=False,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        results = []
        for output, (sample_id, _) in zip(outputs, batch_info):
            text = output[0]["generated_text"]
            score = extract_score(text)
            if score is None:
                if args.verbose:
                    tqdm.write(f"Warning: Could not extract valid score for {sample_id}")
                    tqdm.write(f"Model output: {text[:200]}...")
                score = 0
            results.append(score)
        
        return list(zip(batch_info, results))
    
    except Exception as e:
        tqdm.write(f"Batch processing error: {e}")
        return [(info, 0) for info in batch_info]

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--prediction", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_output_tokens", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    return parser.parse_args()

if __name__ == "__main__":
    args = arguments()
    is_test = args.split.lower() == "test"
    
    print(f"Evaluating {'test' if is_test else 'validation'} set...")
    
    # Load model
    print("Loading model...")
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda",
        temperature=args.temperature,
    )
    
    # Load data
    print("Loading data...")
    reference_data = load_local_data(args.data_root, args.split)
    with open(args.prediction, 'r') as f:
        prediction = json.load(f)
    
    # Prepare batches
    batches = []
    current_batch = []
    total_samples = 0
    
    for data in reference_data:
        sample_id = data["id"]
        if sample_id in prediction:
            current_batch.append((sample_id, data, prediction[sample_id]))
            total_samples += 1
            if len(current_batch) >= args.batch_size:
                batches.append(current_batch)
                current_batch = []
    
    if current_batch:
        batches.append(current_batch)
    
    # Process batches
    print(f"\nProcessing {total_samples} samples...")
    result = defaultdict(list)
    
    with tqdm(total=total_samples, desc="Evaluating") as pbar:
        for batch in batches:
            batch_results = process_batch(pipeline, batch, args, is_test)
            
            for (sample_id, sample_type), score in batch_results:
                result[sample_type].append(score)
                pbar.update(1)
                if score == 0 and args.verbose:
                    tqdm.write(f"Warning: Zero score for {sample_id}")
    
    # Calculate and display results
    print("\nEvaluation Results:")
    total = []
    for sample_type, scores in result.items():
        if scores:
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                score = np.mean(valid_scores)
                print(f"{sample_type.capitalize()} score: {score:.3f}")
                print(f"Valid samples: {len(valid_scores)}/{len(scores)}")
                total.append(score)
            else:
                print(f"Warning: No valid scores for {sample_type}")
    
    if total:
        llm_score = np.mean(total)
        print(f"\nLLM judges: {llm_score:.3f}")
        print(f"Final score ({'test' if is_test else 'validation'} set): {llm_score:.3f}")
        
        if not is_test:
            print("\nCalculating BLEU scores...")
            NLP_HYPOTHESIS = {key: [value.strip()] for key, value in prediction.items()}
            NLP_REFERENCE = {sample["id"]: [get_reference_text(sample)] 
                           for sample in reference_data 
                           if get_reference_text(sample) is not None}
            
            coco_eval = Scorer(NLP_HYPOTHESIS, NLP_REFERENCE)
            total_scores = coco_eval.evaluate()

            for key, value in total_scores.items():
                print(f'{key}: {value:.3f}')

            total_score = llm_score * 0.8 + total_scores["Bleu_3"] * 0.2
            print(f"\nFinal combined score: {total_score:.3f}")
    else:
        print("Error: No valid scores generated.")