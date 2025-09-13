import os
import json
import torch
from tqdm import tqdm

# ANSI color codes for clear terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def check_split_integrity(split_path, expected_embedding_shape):
    """
    Performs a detailed health check on a single data split (train, val, or test).
    """
    split_name = os.path.basename(split_path)
    print(f"\n{Colors.BLUE}--- Checking '{split_name}' Split ---{Colors.ENDC}")

    qa_path = os.path.join(split_path, 'qa.json')
    embeddings_dir = os.path.join(split_path, 'embeddings')

    # --- Pre-flight checks ---
    if not os.path.exists(qa_path):
        print(f"{Colors.RED}[FAIL]{Colors.ENDC} 'qa.json' not found in '{split_path}'")
        return False
    if not os.path.isdir(embeddings_dir):
        print(f"{Colors.RED}[FAIL]{Colors.ENDC} 'embeddings' directory not found in '{split_path}'")
        return False

    with open(qa_path, 'r') as f:
        qa_data = json.load(f)
    
    total_entries = len(qa_data)
    print(f"Found {total_entries} QA entries to verify.")

    # --- Initialize error counters ---
    error_counts = {
        'missing_embedding_file': 0,
        'corrupt_embedding': 0,
        'bad_embedding_shape': 0,
        'nan_in_embedding': 0,
        'inf_in_embedding': 0,
        'zero_embedding': 0,
        'empty_qa_text': 0
    }
    
    is_healthy = True

    for item in tqdm(qa_data, desc=f"Verifying {split_name}"):
        image_name = item.get('image')
        question = item.get('qas', [{}])[0].get('question') # Check first Q for simplicity
        answer = item.get('qas', [{}])[0].get('answer')

        # 1. Check for valid QA text
        if not question or not answer:
            error_counts['empty_qa_text'] += 1
            is_healthy = False
            continue

        # 2. Check for embedding file existence
        base_name = os.path.splitext(image_name)[0]
        embedding_path = os.path.join(embeddings_dir, f"{base_name}.pt")
        if not os.path.exists(embedding_path):
            error_counts['missing_embedding_file'] += 1
            is_healthy = False
            continue

        # 3. Check embedding integrity
        try:
            embedding = torch.load(embedding_path, map_location='cpu')

            if embedding.shape != expected_embedding_shape:
                error_counts['bad_embedding_shape'] += 1
                is_healthy = False
                continue
            if torch.isnan(embedding).any():
                error_counts['nan_in_embedding'] += 1
                is_healthy = False
                continue
            if torch.isinf(embedding).any():
                error_counts['inf_in_embedding'] += 1
                is_healthy = False
                continue
            if torch.all(embedding.eq(0)):
                error_counts['zero_embedding'] += 1
                is_healthy = False
                continue

        except Exception as e:
            error_counts['corrupt_embedding'] += 1
            is_healthy = False
            continue

    # --- Print Summary Report for the Split ---
    print(f"\n{Colors.BLUE}--- Health Check Summary for '{split_name}' ---{Colors.ENDC}")
    if is_healthy:
        print(f"{Colors.GREEN}[PASS]{Colors.ENDC} All {total_entries} entries are healthy!")
    else:
        print(f"{Colors.RED}[FAIL]{Colors.ENDC} Found issues in the dataset:")
        for error_type, count in error_counts.items():
            if count > 0:
                print(f"  - {Colors.YELLOW}{error_type}:{Colors.ENDC} {count} issues")
    
    return is_healthy


def run_full_health_check(dataset_dir):
    """
    Runs the health check on all splits within the main dataset directory.
    """
    print(f"{Colors.BLUE}==========================================={Colors.ENDC}")
    print(f"{Colors.BLUE}  Running Dataset Health Check on '{dataset_dir}'  {Colors.ENDC}")
    print(f"{Colors.BLUE}==========================================={Colors.ENDC}")

    if not os.path.isdir(dataset_dir):
        print(f"{Colors.RED}[FATAL]{Colors.ENDC} Dataset directory not found: '{dataset_dir}'")
        return

    # From our embedding_analysis.json
    expected_shape = torch.Size([768]) 
    
    splits_to_check = ['train', 'val', 'test']
    all_splits_healthy = True

    for split in splits_to_check:
        split_path = os.path.join(dataset_dir, split)
        if os.path.isdir(split_path):
            if not check_split_integrity(split_path, expected_shape):
                all_splits_healthy = False
        else:
            print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} Split directory not found, skipping: '{split_path}'")

    print(f"\n{Colors.BLUE}==========================================={Colors.ENDC}")
    if all_splits_healthy:
        print(f"{Colors.GREEN}  Overall Status: All checks passed! Your dataset is ready for training.  {Colors.ENDC}")
    else:
        print(f"{Colors.RED}  Overall Status: Issues detected. Please review the errors above.  {Colors.ENDC}")
    print(f"{Colors.BLUE}==========================================={Colors.ENDC}")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    # The path to your final, split dataset.
    DATASET_DIRECTORY = 'data/final_split_dataset_versatile'
    
    run_full_health_check(DATASET_DIRECTORY)