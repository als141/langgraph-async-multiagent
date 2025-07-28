#!/usr/bin/env python3
"""
MMLU-Pro Dataset Extraction Script
Extracts 100 balanced questions from MMLU-Pro dataset across all categories.
"""

import csv
import random
from collections import defaultdict
from datasets import load_dataset
import json


def load_mmlu_pro_data():
    """Load MMLU-Pro dataset from HuggingFace"""
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    return dataset


def get_category_distribution(dataset):
    """Analyze category distribution in the dataset"""
    categories = defaultdict(int)
    for split_name in dataset.keys():
        for item in dataset[split_name]:
            categories[item['category']] += 1
    return dict(categories)


def sample_balanced_questions(dataset, target_count=100):
    """
    Sample questions with balanced distribution across categories
    """
    # Combine all splits for sampling
    all_questions = []
    for split_name in dataset.keys():
        for item in dataset[split_name]:
            all_questions.append(item)
    
    # Group by category
    category_questions = defaultdict(list)
    for q in all_questions:
        category_questions[q['category']].append(q)
    
    categories = list(category_questions.keys())
    print(f"Found {len(categories)} categories: {categories}")
    
    # Calculate questions per category for balanced sampling
    questions_per_category = target_count // len(categories)
    remainder = target_count % len(categories)
    
    print(f"Sampling {questions_per_category} questions per category (with {remainder} extra)")
    
    selected_questions = []
    
    for i, category in enumerate(categories):
        available = category_questions[category]
        # Add one extra question for the first 'remainder' categories
        count = questions_per_category + (1 if i < remainder else 0)
        
        if len(available) < count:
            print(f"Warning: Category '{category}' has only {len(available)} questions, taking all")
            selected = available
        else:
            selected = random.sample(available, count)
        
        selected_questions.extend(selected)
        print(f"Category '{category}': selected {len(selected)} questions")
    
    return selected_questions


def format_options_string(options):
    """Format options list into a JSON string for CSV storage"""
    return json.dumps(options, ensure_ascii=False)


def clean_text_for_csv(text):
    """Clean text for CSV storage, handling newlines and quotes"""
    if text is None:
        return ""
    # Replace problematic characters but preserve content
    return str(text).replace('\n', '\\n').replace('\r', '\\r')


def save_to_csv(questions, output_file):
    """Save questions to CSV with specified format"""
    fieldnames = [
        'question_id', 'question', 'options', 'answer', 
        'answer_index', 'cot_content', 'category', 'src', 'question_ja'
    ]
    
    print(f"Saving {len(questions)} questions to {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for q in questions:
            # Prepare row data
            row = {
                'question_id': q.get('question_id', ''),
                'question': clean_text_for_csv(q.get('question', '')),
                'options': format_options_string(q.get('options', [])),
                'answer': clean_text_for_csv(q.get('answer', '')),
                'answer_index': q.get('answer_index', ''),
                'cot_content': clean_text_for_csv(q.get('cot_content', '')),
                'category': q.get('category', ''),
                'src': q.get('src', ''),
                'question_ja': clean_text_for_csv(q.get('question', ''))  # Same as question for now
            }
            writer.writerow(row)


def main():
    """Main extraction process"""
    # Set random seed for reproducible results
    random.seed(42)
    
    # Load dataset
    dataset = load_mmlu_pro_data()
    
    # Analyze distribution
    categories = get_category_distribution(dataset)
    print(f"Category distribution: {categories}")
    
    # Sample balanced questions
    selected_questions = sample_balanced_questions(dataset, target_count=100)
    
    # Shuffle the final list
    random.shuffle(selected_questions)
    
    # Save to CSV
    output_file = '/home/als0028/study/master-research/langgraph-async-multiagent/data/mmlu_pro_100.csv'
    save_to_csv(selected_questions, output_file)
    
    # Print summary
    final_categories = defaultdict(int)
    for q in selected_questions:
        final_categories[q['category']] += 1
    
    print(f"\nFinal distribution:")
    for cat, count in sorted(final_categories.items()):
        print(f"  {cat}: {count} questions")
    
    print(f"\nExtraction complete! Saved to {output_file}")


if __name__ == "__main__":
    main()