import os
import json
import shutil
import random
from tqdm import tqdm

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    YELLOW = '\033[93m'

def split_dataset(master_qa_path, source_embeddings_dir, output_dir, split_ratios=(0.8, 0.1, 0.1), random_seed=42):
    """
    Splits the master QA file and corresponding embeddings into train, validation, and test sets.

    Args:
        master_qa_path (str): The path to the high-quality master QA JSON file.
        source_embeddings_dir (str): The path to the directory containing all original .pt embeddings.
        output_dir (str): The path to the new root directory for the split dataset (e.g., 'final_split_dataset').
        split_ratios (tuple): A tuple with the ratios for (train, val, test).
        random_seed (int): A seed for the random shuffle to ensure reproducibility.
    """
    print(f"{Colors.HEADER}--- Starting High-Quality Dataset Split ---{Colors.ENDC}")
    
    # 1. Load the master QA JSON file, which serves as our master list of images
    try:
        with open(master_qa_path, 'r') as f:
            qa_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Master QA file not found at '{master_qa_path}'")
        return

    # 2. Shuffle the data for a random split. This is crucial for good training.
    random.seed(random_seed)
    random.shuffle(qa_data)
    print(f"Loaded and shuffled {len(qa_data)} total image entries using random seed {random_seed}.")
    
    # 3. Calculate split points based on the ratios
    total_samples = len(qa_data)
    train_end = int(total_samples * split_ratios[0])
    val_end = train_end + int(total_samples * split_ratios[1])
    
    # 4. Create the data splits by slicing the shuffled list
    train_data = qa_data[:train_end]
    val_data = qa_data[train_end:val_end]
    test_data = qa_data[val_end:]
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    # 5. Create directories and process each split
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        print(f"\n{Colors.BLUE}Processing '{split_name}' split with {len(split_data)} image entries...{Colors.ENDC}")
        
        # Create subdirectories for this split (e.g., final_split_dataset/train/embeddings)
        split_output_dir = os.path.join(output_dir, split_name)
        split_embeddings_dir = os.path.join(split_output_dir, 'embeddings')
        os.makedirs(split_embeddings_dir, exist_ok=True)
        
        # Save the qa.json file for this specific split
        split_qa_path = os.path.join(split_output_dir, 'qa.json')
        with open(split_qa_path, 'w') as f:
            json.dump(split_data, f, indent=4)
            
        # Copy the corresponding embedding files from the source to the new destination
        for item in tqdm(split_data, desc=f"Copying {split_name} embeddings"):
            image_name = item['image']
            base_name = os.path.splitext(image_name)[0]
            embedding_filename = f"{base_name}.pt"
            
            source_path = os.path.join(source_embeddings_dir, embedding_filename)
            dest_path = os.path.join(split_embeddings_dir, embedding_filename)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
            else:
                print(f"{Colors.YELLOW}Warning: Embedding file not found for '{image_name}'. Skipping.{Colors.ENDC}")

    print(f"\n{Colors.GREEN}--- Dataset Split Complete! ---{Colors.ENDC}")
    print(f"Train samples (images): {len(train_data)}")
    print(f"Validation samples (images): {len(val_data)}")
    print(f"Test samples (images): {len(test_data)}")
    print(f"Final dataset created at: '{output_dir}'")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # 1. The path to the new, high-quality master QA file you generated.
    MASTER_QA_FILE = 'data/final_qa_master_high_quality.json'
    
    # 2. The source directory where ALL of your original .pt embeddings are stored.
    SOURCE_EMBEDDINGS_DIR = 'data/processed_vlm_dataset/embeddings'
    
    # 3. The name for the new root directory where the final split dataset will be created.
    FINAL_OUTPUT_DIR = 'data/final_split_dataset'
    
    # 4. The desired split ratios (should sum to 1.0).
    #    Default: 80% train, 10% validation, 10% test.
    SPLIT_RATIOS = (0.8, 0.1, 0.1)
    
    # 5. A random seed for reproducibility of the split.
    RANDOM_SEED = 42

    # --- SCRIPT EXECUTION ---
    if not os.path.exists(MASTER_QA_FILE):
        print(f"Error: Master QA file '{MASTER_QA_FILE}' not found.")
        print("Please run 'generate_final_qas.py' first.")
    elif not os.path.isdir(SOURCE_EMBEDDINGS_DIR):
        print(f"Error: Source embeddings directory '{SOURCE_EMBEDDINGS_DIR}' not found.")
    else:
        split_dataset(
            master_qa_path=MASTER_QA_FILE,
            source_embeddings_dir=SOURCE_EMBEDDINGS_DIR,
            output_dir=FINAL_OUTPUT_DIR,
            split_ratios=SPLIT_RATIOS,
            random_seed=RANDOM_SEED
        )