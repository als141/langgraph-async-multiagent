#!/usr/bin/env python3
"""
Check the structure of the original MMLU-Pro dataset.
"""

from datasets import load_dataset

def main():
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    print(f"Dataset splits: {list(dataset.keys())}")
    
    for split_name, split_data in dataset.items():
        print(f"\n=== {split_name.upper()} SPLIT ===")
        print(f"Number of entries: {len(split_data)}")
        
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"Sample entry keys: {list(sample.keys())}")
            print(f"Sample entry types: {[(k, type(v).__name__) for k, v in sample.items()]}")
            
            # Show a sample entry
            print(f"\nSample entry (first in {split_name}):")
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {repr(value[:100])}... (truncated)")
                elif isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {value[:3]}... (showing first 3 of {len(value)})")
                else:
                    print(f"  {key}: {repr(value)}")
    
    # Find the specific questions we're interested in
    target_ids = [95, 7520, 927]
    print(f"\n=== SEARCHING FOR TARGET QUESTIONS {target_ids} ===")
    
    for split_name, split_data in dataset.items():
        found_in_split = []
        for entry in split_data:
            if entry.get('question_id') in target_ids:
                found_in_split.append(entry['question_id'])
        
        if found_in_split:
            print(f"Found in {split_name}: {found_in_split}")

if __name__ == "__main__":
    main()