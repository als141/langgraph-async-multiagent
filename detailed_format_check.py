#!/usr/bin/env python3
"""
Detailed format checking for MMLU-Pro CSV data.
Check for formatting issues with newlines, quotes, special characters.
"""

import json
import pandas as pd
from datasets import load_dataset
import re

def analyze_text_formatting(text, field_name):
    """Analyze text for formatting issues."""
    issues = []
    
    # Check for different types of quotes
    if '"' in text:
        issues.append(f"Contains double quotes")
    if "'" in text:
        issues.append(f"Contains single quotes")
    if "'" in text or "'" in text:
        issues.append(f"Contains curly single quotes")
    if """ in text or """ in text:
        issues.append(f"Contains curly double quotes")
    
    # Check for newlines
    if '\n' in text:
        issues.append(f"Contains newlines")
    if '\r' in text:
        issues.append(f"Contains carriage returns")
    
    # Check for special characters
    if '\\' in text:
        issues.append(f"Contains backslashes")
    if '\t' in text:
        issues.append(f"Contains tabs")
    
    # Check for LaTeX commands
    if '\\text' in text:
        issues.append(f"Contains LaTeX \\text commands")
    if '$' in text:
        issues.append(f"Contains dollar signs (possible LaTeX)")
    
    # Check encoding issues
    non_ascii = [c for c in text if ord(c) > 127]
    if non_ascii:
        unique_chars = list(set(non_ascii))
        issues.append(f"Contains non-ASCII characters: {unique_chars[:10]}...")  # Show first 10
    
    return issues

def detailed_comparison(original, csv_data, question_id):
    """Perform detailed comparison with formatting analysis."""
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS FOR QUESTION {question_id}")
    print(f"{'='*60}")
    
    original_entry = original['data']
    
    # Analyze question text
    print("\n--- QUESTION TEXT ANALYSIS ---")
    orig_question = original_entry.get('question', '')
    csv_question = csv_data.get('question', '')
    
    print(f"Original length: {len(orig_question)}")
    print(f"CSV length: {len(csv_question)}")
    
    orig_issues = analyze_text_formatting(orig_question, "original_question")
    csv_issues = analyze_text_formatting(csv_question, "csv_question")
    
    if orig_issues:
        print(f"Original formatting issues: {orig_issues}")
    if csv_issues:
        print(f"CSV formatting issues: {csv_issues}")
    
    # Character-by-character comparison if they differ
    if orig_question != csv_question:
        print("❌ Question texts differ!")
        print(f"First difference at position: {next((i for i, (a, b) in enumerate(zip(orig_question, csv_question)) if a != b), min(len(orig_question), len(csv_question)))}")
    else:
        print("✓ Question texts are identical")
    
    # Analyze options
    print("\n--- OPTIONS ANALYSIS ---")
    orig_options = original_entry.get('options', [])
    csv_options_str = csv_data.get('options', '')
    
    print(f"Original options count: {len(orig_options)}")
    print(f"CSV options string length: {len(csv_options_str)}")
    
    # Parse CSV options
    try:
        csv_options = json.loads(csv_options_str)
        print(f"Parsed CSV options count: {len(csv_options)}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return
    
    # Analyze each option
    for i, (orig_opt, csv_opt) in enumerate(zip(orig_options, csv_options)):
        print(f"\n  Option {i}:")
        print(f"    Original: {repr(orig_opt)}")
        print(f"    CSV:      {repr(csv_opt)}")
        
        if orig_opt != csv_opt:
            print(f"    ❌ Options differ!")
            orig_opt_issues = analyze_text_formatting(orig_opt, f"original_option_{i}")
            csv_opt_issues = analyze_text_formatting(csv_opt, f"csv_option_{i}")
            
            if orig_opt_issues:
                print(f"    Original issues: {orig_opt_issues}")
            if csv_opt_issues:
                print(f"    CSV issues: {csv_opt_issues}")
        else:
            print(f"    ✓ Identical")
    
    # Check JSON structure validity
    print("\n--- JSON STRUCTURE CHECK ---")
    try:
        # Re-encode and decode to check for any issues
        re_encoded = json.dumps(csv_options)
        re_parsed = json.loads(re_encoded)
        if csv_options == re_parsed:
            print("✓ JSON structure is valid and stable")
        else:
            print("❌ JSON structure has issues after re-encoding")
    except Exception as e:
        print(f"❌ JSON structure error: {e}")
    
    # Analyze answer correspondence
    print("\n--- ANSWER CORRESPONDENCE ---")
    answer_index = csv_data.get('answer_index')
    answer_letter = csv_data.get('answer')
    
    if answer_index is not None and 0 <= answer_index < len(csv_options):
        corresponding_option = csv_options[answer_index]
        print(f"Answer letter: {answer_letter}")
        print(f"Answer index: {answer_index}")
        print(f"Corresponding option: {repr(corresponding_option)}")
        
        # Check if the answer makes sense
        expected_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        if answer_index < len(expected_letters):
            expected_letter = expected_letters[answer_index]
            if answer_letter == expected_letter:
                print(f"✓ Answer letter {answer_letter} correctly corresponds to index {answer_index}")
            else:
                print(f"❌ Answer letter {answer_letter} doesn't match expected {expected_letter} for index {answer_index}")
    
    print(f"\n{'='*60}")

def main():
    question_ids = [95, 7520, 927]
    csv_path = "/home/als0028/study/master-research/langgraph-async-multiagent/data/mmlu_pro_100.csv"
    
    try:
        # Load datasets
        print("Loading original dataset...")
        original_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        
        # Find questions in original
        original_questions = {}
        for split_name, split_data in original_dataset.items():
            for idx, entry in enumerate(split_data):
                if entry.get('question_id') in question_ids:
                    question_id = entry['question_id']
                    original_questions[question_id] = {
                        'split': split_name,
                        'index': idx,
                        'data': entry
                    }
        
        # Load CSV data
        print("Loading CSV data...")
        df = pd.read_csv(csv_path)
        csv_questions = {}
        for _, row in df.iterrows():
            if row['question_id'] in question_ids:
                csv_questions[row['question_id']] = row.to_dict()
        
        # Perform detailed analysis
        for question_id in question_ids:
            if question_id in original_questions and question_id in csv_questions:
                detailed_comparison(
                    original_questions[question_id],
                    csv_questions[question_id],
                    question_id
                )
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()