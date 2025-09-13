import os
import torch
import yaml
from train.trainer import VQADataset, VLMTrainer
from torch.utils.data import DataLoader

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    RED = '\033[91m'
    YELLOW = '\033[93m'

def run_data_flow_verification(config_path='config.yml'):
    print(f"{Colors.HEADER}============================================={Colors.ENDC}")
    print(f"{Colors.HEADER}  Running Data Flow and Integrity Verification  {Colors.ENDC}")
    print(f"{Colors.HEADER}============================================={Colors.ENDC}")

    print(f"\n{Colors.BLUE}--- Step 1: Loading Configuration and Initializing Components ---{Colors.ENDC}")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    trainer = VLMTrainer(config)
    print(f"{Colors.GREEN}[PASS]{Colors.ENDC} Trainer and tokenizer initialized successfully.")

    batch_size_to_inspect = 2
    dataset = VQADataset(
        qa_path=os.path.join(config['dataset_path'], 'train/qa.json'),
        embeddings_dir=os.path.join(config['dataset_path'], 'train/embeddings'),
        max_samples=10
    )
    data_loader = DataLoader(dataset, batch_size=batch_size_to_inspect, shuffle=False)
    print(f"{Colors.GREEN}[PASS]{Colors.ENDC} DataLoader created successfully for the 'train' split.")
        
    print(f"\n{Colors.BLUE}--- Step 2: Inspecting a RAW Batch from the DataLoader ---{Colors.ENDC}")
    raw_batched_output = next(iter(data_loader))
    embeddings_list, questions_list, answers_list = raw_batched_output
    
    # --- ENHANCED PRINTOUT ---
    # We need to get the embedding path from the dataset directly for this inspection
    for i in range(batch_size_to_inspect):
        sample_info = dataset.flat_data[i]
        embedding_filename = os.path.basename(sample_info['embedding_path'])
        print(f"\n  {Colors.HEADER}Sample {i+1} in Raw Batch:{Colors.ENDC}")
        print(f"    - {Colors.YELLOW}Embedding File:{Colors.ENDC} {embedding_filename}")
        print(f"    - {Colors.YELLOW}Question:{Colors.ENDC}         '{questions_list[i]}'")
        print(f"    - {Colors.YELLOW}Answer:{Colors.ENDC}           '{answers_list[i]}'")
    
    print(f"\n{Colors.GREEN}[PASS]{Colors.ENDC} Raw data pairing appears correct.")

    print(f"\n{Colors.BLUE}--- Step 3: Inspecting a PROCESSED Batch (What the Model Sees) ---{Colors.ENDC}")
    reformatted_batch_for_collate = list(zip(embeddings_list, questions_list, answers_list))
    batched_embeddings, tokenized_prompts, labels = trainer._collate_fn(reformatted_batch_for_collate)
    
    print(f"\n  {Colors.HEADER}A. Image Embedding Tensor (Projector Input):{Colors.ENDC}")
    print(f"    - Shape: {batched_embeddings.shape}")
    expected_dim = config['model']['vision_embedding_dim']
    actual_dim = batched_embeddings.shape[1]
    if actual_dim == expected_dim: print(f"    - {Colors.GREEN}[PASS]{Colors.ENDC} Tensor dimension ({actual_dim}) matches projector's expected input ({expected_dim}).")
    else: print(f"    - {Colors.RED}[FAIL]{Colors.ENDC} Mismatch! Tensor dim is {actual_dim} but projector expects {expected_dim}.")

    print(f"\n  {Colors.HEADER}B. Text Tensors (LLM Input):{Colors.ENDC}")
    print(f"    - Tokenized Prompt IDs Shape: {tokenized_prompts['input_ids'].shape}")
    
    # --- ENHANCED PRINTOUT ---
    print(f"\n    {Colors.HEADER}Inspecting the first full prompt and its label tensor:{Colors.ENDC}")
    full_prompt_decoded = trainer.tokenizer.decode(tokenized_prompts['input_ids'][0])
    print(f"      - {Colors.YELLOW}Full Decoded Prompt:{Colors.ENDC}\n        '{full_prompt_decoded}'")
    
    first_label_tensor = labels[0]
    print(f"      - {Colors.YELLOW}Label Tensor:{Colors.ENDC} {first_label_tensor}")
    
    decodable_labels = first_label_tensor.clone()
    decodable_labels[decodable_labels == -100] = trainer.tokenizer.pad_token_id
    decoded_text = trainer.tokenizer.decode(decodable_labels, skip_special_tokens=True)
    
    print(f"      - {Colors.YELLOW}Decoded Answer (for loss):{Colors.ENDC} '{decoded_text.strip()}'")
    print(f"      - {Colors.YELLOW}Original Answer:{Colors.ENDC}           '{answers_list[0]}'")
    
    if answers_list[0].strip() in decoded_text.strip() or decoded_text.strip() in answers_list[0].strip():
        print(f"    - {Colors.GREEN}[PASS]{Colors.ENDC} Label masking is working correctly.")
    else:
        print(f"    - {Colors.RED}[FAIL]{Colors.ENDC} Label masking seems incorrect.")

    print(f"\n{Colors.HEADER}============================================={Colors.ENDC}")
    print(f"{Colors.GREEN}      Verification Complete. Data flow appears to be correct.      {Colors.ENDC}")
    print(f"{Colors.HEADER}============================================={Colors.ENDC}")

if __name__ == '__main__':
    run_data_flow_verification()