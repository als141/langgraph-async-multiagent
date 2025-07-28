#!/usr/bin/env python3
"""
Verify extracted MMLU-Pro CSV data against the original HuggingFace dataset.
"""

import json
import pandas as pd
from datasets import load_dataset
import ast

def load_original_dataset():
    """Load the original MMLU-Pro dataset from HuggingFace."""
    print("Loading MMLU-Pro dataset from HuggingFace...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    return dataset

def find_questions_in_original(dataset, question_ids):
    """Find specific questions by ID in the original dataset."""
    found_questions = {}
    
    # Search through all splits
    for split_name, split_data in dataset.items():
        print(f"Searching in {split_name} split ({len(split_data)} entries)...")
        
        for idx, entry in enumerate(split_data):
            if entry.get('question_id') in question_ids:
                question_id = entry['question_id']
                found_questions[question_id] = {
                    'split': split_name,
                    'index': idx,
                    'data': entry
                }
                print(f"Found question_id {question_id} in {split_name} at index {idx}")
    
    return found_questions

def load_csv_data(csv_path, question_ids):
    """Load specific questions from the CSV file."""
    print(f"Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    csv_questions = {}
    for _, row in df.iterrows():
        if row['question_id'] in question_ids:
            csv_questions[row['question_id']] = row.to_dict()
    
    return csv_questions

def parse_options(options_str):
    """Parse the options string from CSV format."""
    try:
        # Try to parse as JSON array
        return json.loads(options_str)
    except json.JSONDecodeError:
        try:
            # Try to parse as Python literal
            return ast.literal_eval(options_str)
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse options: {options_str}")
            return None

def compare_questions(original_data, csv_data, question_id):
    """Compare a single question between original and CSV data."""
    print(f"\n=== Comparing Question ID {question_id} ===")
    
    original = original_data['data']
    csv_row = csv_data
    
    discrepancies = []
    
    # Compare question_id
    if original.get('question_id') != csv_row.get('question_id'):
        discrepancies.append(f"question_id mismatch: original={original.get('question_id')}, csv={csv_row.get('question_id')}")
    else:
        print("✓ question_id matches")
    
    # Compare question text
    original_question = original.get('question', '').strip()
    csv_question = csv_row.get('question', '').strip()
    
    if original_question != csv_question:
        discrepancies.append(f"Question text mismatch")
        print(f"Original: {original_question}")
        print(f"CSV:      {csv_question}")
    else:
        print("✓ Question text matches")
    
    # Compare options
    original_options = original.get('options', [])
    csv_options_str = csv_row.get('options', '')
    csv_options = parse_options(csv_options_str)
    
    if csv_options is None:
        discrepancies.append("Could not parse CSV options")
    elif original_options != csv_options:
        discrepancies.append("Options mismatch")
        print(f"Original options ({len(original_options)}): {original_options}")
        print(f"CSV options ({len(csv_options) if csv_options else 'None'}): {csv_options}")
        
        # Check individual options
        if csv_options and len(original_options) == len(csv_options):
            for i, (orig, csv_opt) in enumerate(zip(original_options, csv_options)):
                if orig != csv_opt:
                    print(f"  Option {i} differs:")
                    print(f"    Original: {orig}")
                    print(f"    CSV:      {csv_opt}")
    else:
        print("✓ Options match")
    
    # Compare answer
    original_answer = original.get('answer')
    csv_answer = csv_row.get('answer')
    
    if original_answer != csv_answer:
        discrepancies.append(f"Answer mismatch: original={original_answer}, csv={csv_answer}")
    else:
        print("✓ Answer matches")
    
    # Compare answer_index
    original_answer_index = original.get('answer_index')
    csv_answer_index = csv_row.get('answer_index')
    
    if original_answer_index != csv_answer_index:
        discrepancies.append(f"Answer index mismatch: original={original_answer_index}, csv={csv_answer_index}")
    else:
        print("✓ Answer index matches")
    
    # Verify answer_index corresponds to correct option
    if csv_options and csv_answer_index is not None:
        try:
            if 0 <= csv_answer_index < len(csv_options):
                expected_answer_option = csv_options[csv_answer_index]
                print(f"✓ Answer index {csv_answer_index} points to: {expected_answer_option}")
            else:
                discrepancies.append(f"Answer index {csv_answer_index} out of range for {len(csv_options)} options")
        except (TypeError, IndexError) as e:
            discrepancies.append(f"Error verifying answer index: {e}")
    
    # Compare category
    original_category = original.get('category')
    csv_category = csv_row.get('category')
    
    if original_category != csv_category:
        discrepancies.append(f"Category mismatch: original={original_category}, csv={csv_category}")
    else:
        print("✓ Category matches")
    
    # Compare src
    original_src = original.get('src')
    csv_src = csv_row.get('src')
    
    if original_src != csv_src:
        discrepancies.append(f"Src mismatch: original={original_src}, csv={csv_src}")
    else:
        print("✓ Src matches")
    
    # Compare cot_content
    original_cot = original.get('cot_content')
    csv_cot = csv_row.get('cot_content')
    
    # Handle NaN values in CSV
    if pd.isna(csv_cot):
        csv_cot = None
    
    if original_cot != csv_cot:
        if original_cot == '' and csv_cot is None:
            print("✓ CoT content matches (both empty)")
        else:
            discrepancies.append(f"CoT content mismatch")
            print(f"Original CoT: {repr(original_cot)}")
            print(f"CSV CoT: {repr(csv_cot)}")
    else:
        print("✓ CoT content matches")
    
    return discrepancies

def main():
    # Question IDs to verify
    question_ids = [95, 7520, 927]
    csv_path = "/home/als0028/study/master-research/langgraph-async-multiagent/data/mmlu_pro_100.csv"
    
    try:
        # Load original dataset
        original_dataset = load_original_dataset()
        
        # Find questions in original dataset
        original_questions = find_questions_in_original(original_dataset, question_ids)
        
        # Load CSV data
        csv_questions = load_csv_data(csv_path, question_ids)
        
        print(f"\nFound {len(original_questions)} questions in original dataset")
        print(f"Found {len(csv_questions)} questions in CSV data")
        
        # Compare each question
        all_discrepancies = {}
        
        for question_id in question_ids:
            if question_id in original_questions and question_id in csv_questions:
                discrepancies = compare_questions(
                    original_questions[question_id],
                    csv_questions[question_id],
                    question_id
                )
                if discrepancies:
                    all_discrepancies[question_id] = discrepancies
            else:
                print(f"\nQuestion {question_id} not found in both datasets:")
                print(f"  In original: {question_id in original_questions}")
                print(f"  In CSV: {question_id in csv_questions}")
        
        # Summary
        print("\n" + "="*50)
        print("VERIFICATION SUMMARY")
        print("="*50)
        
        if not all_discrepancies:
            print("✅ All questions match perfectly!")
        else:
            print("❌ Found discrepancies:")
            for question_id, discrepancies in all_discrepancies.items():
                print(f"\nQuestion {question_id}:")
                for discrepancy in discrepancies:
                    print(f"  - {discrepancy}")
        
        # Test JSON parsing of options
        print("\n" + "="*50)
        print("JSON PARSING TEST")
        print("="*50)
        
        for question_id in question_ids:
            if question_id in csv_questions:
                options_str = csv_questions[question_id]['options']
                parsed_options = parse_options(options_str)
                if parsed_options:
                    print(f"✓ Question {question_id}: Successfully parsed {len(parsed_options)} options")
                else:
                    print(f"❌ Question {question_id}: Failed to parse options")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()