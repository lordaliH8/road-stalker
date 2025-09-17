import os
import json
from collections import defaultdict

# ANSI color codes for clear terminal output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    YELLOW = '\033[93m'

def categorize_question(question_text):
    """Categorizes a question based on its starting phrase."""
    question_lower = question_text.lower()
    if question_lower.startswith("what are the objects"):
        return "Objects Listing"
    elif question_lower.startswith("what is the color"):
        return "Traffic Light Color"
    elif question_lower.startswith("is there a"):
        return "Object Presence"
    else:
        return "Other"

def analyze_split(split_path):
    """Analyzes the qa.json file for a single data split."""
    split_name = os.path.basename(split_path)
    print(f"\n{Colors.HEADER}--- Analyzing Split: {split_name.upper()} ---{Colors.ENDC}")
    
    qa_path = os.path.join(split_path, 'qa.json')
    if not os.path.exists(qa_path):
        print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} 'qa.json' not found. Skipping.")
        return

    with open(qa_path, 'r') as f:
        qa_data = json.load(f)

    all_qa_pairs = []
    question_type_counts = defaultdict(int)

    for item in qa_data:
        for qa_pair in item.get('qas', []):
            question = qa_pair['question']
            answer = qa_pair['answer']
            
            # Add to list for redundancy check
            all_qa_pairs.append((question, answer))
            
            # Categorize and count
            category = categorize_question(question)
            question_type_counts[category] += 1

    total_pairs = len(all_qa_pairs)
    unique_pairs = len(set(all_qa_pairs))
    duplicate_pairs = total_pairs - unique_pairs

    print(f"{Colors.BLUE}Redundancy Check:{Colors.ENDC}")
    print(f"  - Total QA Pairs:    {total_pairs}")
    print(f"  - Unique QA Pairs:   {unique_pairs}")
    print(f"  - Duplicate Pairs:   {duplicate_pairs} ({((duplicate_pairs/total_pairs)*100):.2f}% duplicates)")

    print(f"\n{Colors.BLUE}Question Type Distribution:{Colors.ENDC}")
    for q_type, count in sorted(question_type_counts.items()):
        print(f"  - {q_type:<20}: {count:>6} pairs ({((count/total_pairs)*100):.2f}%)")

def run_full_qa_analysis(dataset_dir):
    """Runs the analysis on all splits."""
    print(f"{Colors.HEADER}======================================={Colors.ENDC}")
    print(f"{Colors.HEADER}  Running Full QA Dataset Analysis on '{dataset_dir}'  {Colors.ENDC}")
    print(f"{Colors.HEADER}======================================={Colors.ENDC}")
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_dir, split)
        analyze_split(split_path)
        
    print(f"\n{Colors.HEADER}======================================={Colors.ENDC}")

if __name__ == '__main__':
    DATASET_DIRECTORY = 'data/final_split_dataset'
    run_full_qa_analysis(DATASET_DIRECTORY)