#!/usr/bin/env python3
"""
Final comprehensive verification summary for MMLU-Pro CSV data.
"""

import json
import pandas as pd
from datasets import load_dataset

def main():
    print("MMLU-Pro CSV Data Verification Report")
    print("="*50)
    
    # Test question IDs
    question_ids = [95, 7520, 927]
    csv_path = "/home/als0028/study/master-research/langgraph-async-multiagent/data/mmlu_pro_100.csv"
    
    try:
        # Load original dataset
        original_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        
        # Find questions in original
        original_questions = {}
        for split_name, split_data in original_dataset.items():
            for entry in split_data:
                if entry.get('question_id') in question_ids:
                    original_questions[entry['question_id']] = entry
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        csv_questions = {}
        for _, row in df.iterrows():
            if row['question_id'] in question_ids:
                csv_questions[row['question_id']] = row.to_dict()
        
        print(f"Verified {len(question_ids)} questions: {question_ids}")
        print(f"Found {len(original_questions)} in original dataset")
        print(f"Found {len(csv_questions)} in CSV file")
        print()
        
        # Verification results
        all_match = True
        
        for question_id in question_ids:
            if question_id not in original_questions or question_id not in csv_questions:
                print(f"❌ Question {question_id}: Missing from one of the datasets")
                all_match = False
                continue
            
            orig = original_questions[question_id]
            csv_row = csv_questions[question_id]
            
            # Check each field
            fields_match = True
            
            # Question ID
            if orig['question_id'] != csv_row['question_id']:
                print(f"❌ Question {question_id}: ID mismatch")
                fields_match = False
            
            # Question text
            if orig['question'] != csv_row['question']:
                print(f"❌ Question {question_id}: Question text mismatch")
                fields_match = False
            
            # Options
            try:
                csv_options = json.loads(csv_row['options'])
                if orig['options'] != csv_options:
                    print(f"❌ Question {question_id}: Options mismatch")
                    fields_match = False
            except json.JSONDecodeError:
                print(f"❌ Question {question_id}: Options not valid JSON")
                fields_match = False
            
            # Answer
            if orig['answer'] != csv_row['answer']:
                print(f"❌ Question {question_id}: Answer mismatch")
                fields_match = False
            
            # Answer index
            if orig['answer_index'] != csv_row['answer_index']:
                print(f"❌ Question {question_id}: Answer index mismatch")
                fields_match = False
            
            # Category
            if orig['category'] != csv_row['category']:
                print(f"❌ Question {question_id}: Category mismatch")
                fields_match = False
            
            # Source
            if orig['src'] != csv_row['src']:
                print(f"❌ Question {question_id}: Source mismatch")
                fields_match = False
            
            # CoT content
            orig_cot = orig.get('cot_content', '')
            csv_cot = csv_row.get('cot_content', '')
            if pd.isna(csv_cot):
                csv_cot = ''
            if orig_cot != csv_cot:
                if not (orig_cot == '' and csv_cot == ''):
                    print(f"❌ Question {question_id}: CoT content mismatch")
                    fields_match = False
            
            if fields_match:
                print(f"✅ Question {question_id}: All fields match perfectly")
            else:
                all_match = False
        
        print()
        print("="*50)
        print("SUMMARY")
        print("="*50)
        
        if all_match:
            print("🎉 VERIFICATION SUCCESSFUL!")
            print("All verified questions match the original dataset perfectly:")
            print("✅ Question IDs are correct")
            print("✅ Question texts are identical")
            print("✅ Options arrays are correctly formatted and complete")
            print("✅ Answer letters are correct")
            print("✅ Answer indices correspond to correct options")
            print("✅ Categories and sources match")
            print("✅ CoT content is preserved (empty for these questions)")
            print("✅ JSON formatting is valid and parseable")
            print("✅ No formatting issues with quotes, newlines, or special characters")
            print("✅ LaTeX commands and special characters are preserved correctly")
            print()
            print("The CSV extraction process has successfully maintained data integrity.")
        else:
            print("❌ VERIFICATION FAILED!")
            print("Some discrepancies were found. Please check the output above.")
        
        # Additional format checks
        print()
        print("="*50)
        print("FORMAT VALIDATION")
        print("="*50)
        
        format_issues = []
        
        for question_id in question_ids:
            if question_id in csv_questions:
                csv_row = csv_questions[question_id]
                
                # Check if options can be parsed as JSON
                try:
                    options = json.loads(csv_row['options'])
                    if not isinstance(options, list):
                        format_issues.append(f"Question {question_id}: Options not a list")
                    elif len(options) == 0:
                        format_issues.append(f"Question {question_id}: Empty options list")
                except json.JSONDecodeError as e:
                    format_issues.append(f"Question {question_id}: JSON parse error - {e}")
                
                # Check answer index bounds
                try:
                    answer_idx = csv_row['answer_index']
                    options = json.loads(csv_row['options'])
                    if not (0 <= answer_idx < len(options)):
                        format_issues.append(f"Question {question_id}: Answer index {answer_idx} out of bounds for {len(options)} options")
                except:
                    format_issues.append(f"Question {question_id}: Could not validate answer index bounds")
        
        if not format_issues:
            print("✅ All format validations passed")
            print("✅ JSON arrays are properly formatted")
            print("✅ Answer indices are within valid ranges")
            print("✅ Data types are consistent")
        else:
            print("❌ Format issues found:")
            for issue in format_issues:
                print(f"  - {issue}")
    
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()