import os
import json
import shutil
import random
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from collections import defaultdict

# --- ANSI color codes ---
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

# --- Part 1: QA Generation and Analysis Logic ---
def generate_and_analyze_qas(image_metadata, num_presence_questions=6):
    print(f"\n{Colors.BLUE}Step 2: Generating and analyzing high-quality QA pairs...{Colors.ENDC}")
    all_objects = set()
    for item in image_metadata:
        for obj in item.get('objects', []): all_objects.add(obj)
    print(f"Found {len(all_objects)} unique object categories in the source data.")
    versatile_qa_list = []
    weather_templates = ["How is the weather in the image?", "What's the weather like in this scene?"]
    color_templates = ["What is the traffic light's color in the image?", "What color is the traffic light showing?"]
    presence_templates = ["Is there a {} visible in the image?", "Does the image contain a {}?"]
    for image_info in tqdm(image_metadata, desc="Generating & Sampling QAs"):
        qas = []
        weather = image_info.get("weather")
        if weather and weather not in ['undefined', 'none']: qas.append({"question": random.choice(weather_templates), "answer": weather})
        traffic_lights = image_info.get("traffic_lights", [])
        question_color = random.choice(color_templates)
        if not traffic_lights: qas.append({"question": question_color, "answer": "there is no traffic light in this image"})
        else:
            colors = {light['color'] for light in traffic_lights if light.get('color')}
            if len(colors) == 1:
                answer = colors.pop()
                if answer != 'none': qas.append({"question": question_color, "answer": answer})
        present_objects = set(image_info.get("objects", []))
        absent_objects = all_objects - present_objects
        num_positive = num_presence_questions // 2; num_negative = num_presence_questions - num_positive
        if present_objects:
            positive_samples = random.sample(list(present_objects), min(num_positive, len(present_objects)))
            for obj in positive_samples: qas.append({"question": random.choice(presence_templates).format(obj), "answer": "yes"})
        if absent_objects:
            negative_samples = random.sample(list(absent_objects), min(num_negative, len(absent_objects)))
            for obj in negative_samples: qas.append({"question": random.choice(presence_templates).format(obj), "answer": "no"})
        if qas:
            chosen_qa = random.choice(qas)
            versatile_qa_list.append({"image": image_info['name'], "question": chosen_qa['question'], "answer": chosen_qa['answer']})
    print(f"Created a versatile list with {len(versatile_qa_list)} unique image-QA pairs.")
    return versatile_qa_list

# --- Part 2: Embedding Generation Logic ---
def create_embeddings(image_paths, output_dir, model_name='openai/clip-vit-base-patch32'):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{Colors.BLUE}Creating embeddings using device: {device}{Colors.ENDC}")
    model = CLIPVisionModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Generating new embeddings"):
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            output_path = os.path.join(output_dir, f"{base_name}.pt")
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                embedding = outputs.pooler_output.squeeze().cpu()
                torch.save(embedding, output_path)
            except Exception as e:
                print(f"\n{Colors.YELLOW}Warning: Could not process image {image_name}. Skipping. Error: {e}{Colors.ENDC}")

# --- NEW: SIMPLIFIED SPLITTING LOGIC ---
def save_split(split_name, data, source_embeddings_dir, output_dir):
    """Saves a single split (e.g., 'train') to its final directory."""
    print(f"\n{Colors.BLUE}Processing '{split_name}' split with {len(data)} image entries...{Colors.ENDC}")
    
    split_dir = os.path.join(output_dir, split_name)
    split_embed_dir = os.path.join(split_dir, 'embeddings')
    os.makedirs(split_embed_dir, exist_ok=True)
    
    # Format the data to match what the VQADataset class expects
    formatted_data = [{"image": item["image"], "qas": [{"question": item["question"], "answer": item["answer"]}]} for item in data]
    with open(os.path.join(split_dir, 'qa.json'), 'w') as f:
        json.dump(formatted_data, f, indent=4)
        
    for item in tqdm(data, desc=f"Copying {split_name} embeddings"):
        base_name = os.path.splitext(item['image'])[0]
        source_path = os.path.join(source_embeddings_dir, f"{base_name}.pt")
        dest_path = os.path.join(split_embed_dir, f"{base_name}.pt")
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
        else:
            print(f"{Colors.YELLOW}Warning: Embedding for {item['image']} not found, skipping copy.{Colors.ENDC}")


# --- Main Orchestrator ---
def build_dataset_pipeline(config):
    print(f"{Colors.HEADER}====================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}  Starting Versatile Dataset Build Pipeline (1 QA/Image)  {Colors.ENDC}")
    print(f"{Colors.HEADER}====================================================={Colors.ENDC}")

    print(f"\n{Colors.BLUE}Step 1: Loading and filtering source labels...{Colors.ENDC}")
    with open(config['source_files']['simplified_labels'], 'r') as f: simplified_data = json.load(f)
    all_images_metadata = simplified_data.get('bdd100k_labels_images_train.json', []) + simplified_data.get('bdd100k_labels_images_val.json', [])
    image_files_on_disk = {f for root, _, files in os.walk(config['paths']['raw_image_dir']) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    unique_image_metadata = {item['name']: item for item in all_images_metadata if item['name'] in image_files_on_disk}.values()
    print(f"Found {len(image_files_on_disk)} images on disk, using {len(unique_image_metadata)} valid metadata entries.")
    
    versatile_qa_list = generate_and_analyze_qas(unique_image_metadata, config['qa_generation']['presence_questions_per_image'])

    random.seed(config['random_seed'])
    random.shuffle(versatile_qa_list)
    final_qa_samples = versatile_qa_list[:config['total_samples']]
    print(f"\n{Colors.BLUE}Step 3: Selected a random subset of {len(final_qa_samples)} pairs for the final dataset.{Colors.ENDC}")

    print(f"\n{Colors.BLUE}Step 4: Performing fresh analysis on the final {len(final_qa_samples)} samples...{Colors.ENDC}")
    final_analysis = {"dataset_summary": {"total_qa_pairs": len(final_qa_samples)}, "question_type_distribution": defaultdict(int), "answer_distribution": defaultdict(int)}
    for item in final_qa_samples:
        q_lower = item['question'].lower()
        if "weather" in q_lower: final_analysis["question_type_distribution"]["Weather"] += 1
        elif "color" in q_lower: final_analysis["question_type_distribution"]["Traffic Light Color"] += 1
        elif "visible" in q_lower or "contain" in q_lower: final_analysis["question_type_distribution"]["Object Presence"] += 1
        final_analysis["answer_distribution"][item['answer']] += 1
    analysis_output_path = config['source_files']['analysis_output']
    with open(analysis_output_path, 'w') as f: json.dump(final_analysis, f, indent=4)
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} New analysis saved to '{analysis_output_path}'.")

    embeddings_dir = config['paths']['embeddings_dir']
    print(f"\n{Colors.BLUE}Step 5: Cleaning and re-creating embeddings for {len(final_qa_samples)} images...{Colors.ENDC}")
    if os.path.exists(embeddings_dir): shutil.rmtree(embeddings_dir)
    os.makedirs(embeddings_dir)
    image_path_map = {f: os.path.join(root, f) for root, _, files in os.walk(config['paths']['raw_image_dir']) for f in files}
    image_paths_to_embed = [image_path_map[item['image']] for item in final_qa_samples if item['image'] in image_path_map]
    create_embeddings(image_paths_to_embed, embeddings_dir)
    
    final_dataset_dir = config['paths']['final_split_dir']
    print(f"\n{Colors.BLUE}Step 6: Splitting final dataset into '{final_dataset_dir}'...{Colors.ENDC}")
    if os.path.exists(final_dataset_dir): shutil.rmtree(final_dataset_dir)
    
    # --- CORRECTED SPLITTING LOGIC ---
    train_count = config['split_counts']['train']
    val_count = config['split_counts']['val']
    train_data = final_qa_samples[:train_count]
    val_data = final_qa_samples[train_count : train_count + val_count]
    test_data = final_qa_samples[train_count + val_count:]
    
    save_split('train', train_data, embeddings_dir, final_dataset_dir)
    save_split('val', val_data, embeddings_dir, final_dataset_dir)
    save_split('test', test_data, embeddings_dir, final_dataset_dir)

    print(f"\n{Colors.GREEN}==========================================={Colors.ENDC}")
    print(f"{Colors.GREEN}      Versatile Dataset Build Complete!    {Colors.ENDC}")
    print(f"      Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"{Colors.GREEN}==========================================={Colors.ENDC}")

if __name__ == '__main__':
    build_config = {
        "source_files": {"simplified_labels": "data/simplified_labels_for_vlm.json", "analysis_output": "data/final_versatile_dataset_analysis.json"},
        "paths": {"raw_image_dir": "data/bdd100k/bdd100k/bdd100k/images/100k", "embeddings_dir": "data/processed_vlm_dataset_versatile/embeddings", "final_split_dir": "data/final_split_dataset_versatile"},
        "qa_generation": {"presence_questions_per_image": 6},
        "total_samples": 15000,
        "split_counts": { "train": 12000, "val": 2000, "test": 1000 },
        "random_seed": 42
    }
    if build_config['paths']['raw_image_dir'] == 'path/to/your/images':
        print(f"{Colors.RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{Colors.ENDC}")
        print(f"{Colors.RED}!!! PLEASE UPDATE the 'raw_image_dir' path in the script    !!!{Colors.ENDC}")
        print(f"{Colors.RED}!!! to point to your main image folder before running.      !!!{Colors.ENDC}")
        print(f"{Colors.RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{Colors.ENDC}")
    else:
        build_dataset_pipeline(build_config)