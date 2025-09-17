import os
import json
import random
from tqdm import tqdm
from collections import defaultdict

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'

def generate_balanced_qas_for_image(image_info, all_possible_objects, num_presence_questions=6):
    """
    Generates a high-quality, balanced, and non-redundant set of QA pairs
    for a single image based on the new, improved rules.
    """
    qas = []
    
    # --- Prompt Templates for Variety ---
    weather_templates = [
        "How is the weather in the image?",
        "What's the weather like in this scene?",
        "Describe the weather conditions."
    ]
    color_templates = [
        "What is the traffic light's color in the image?",
        "What color is the traffic light showing?",
        "Can you identify the color of the traffic light?"
    ]
    presence_templates = [
        "Is there a {} visible in the image?",
        "Does the image contain a {}?",
        "Can you see a {} in this picture?"
    ]

    # --- 1. Weather Question ---
    weather = image_info.get("weather")
    if weather and weather not in ['undefined', 'none']:
        question = random.choice(weather_templates)
        qas.append({"question": question, "answer": weather})

    # --- 2. Traffic Light Question (with new ambiguity logic) ---
    traffic_lights = image_info.get("traffic_lights", [])
    question = random.choice(color_templates)
    
    if not traffic_lights:
        # Case 1: No traffic lights in the image
        answer = "there is no traffic light in this image"
        qas.append({"question": question, "answer": answer})
    else:
        # Case 2: Traffic lights are present, check for color consistency
        colors = {light['color'] for light in traffic_lights if light.get('color')}
        if len(colors) == 1:
            # Only one unique color, this is a valid sample
            answer = colors.pop()
            if answer != 'none':
                qas.append({"question": question, "answer": answer})
        # If len(colors) is 0 or > 1 (no colors or mixed colors), we skip this question.

    # --- 3. Object Presence Questions (Balanced Yes/No) ---
    present_objects = set(image_info.get("objects", []))
    absent_objects = all_possible_objects - present_objects
    
    num_positive = num_presence_questions // 2
    num_negative = num_presence_questions - num_positive

    # Generate positive ("yes") samples
    if present_objects:
        positive_samples = random.sample(list(present_objects), min(num_positive, len(present_objects)))
        for obj in positive_samples:
            question = random.choice(presence_templates).format(obj)
            qas.append({"question": question, "answer": "yes"})
            
    # Generate negative ("no") samples
    if absent_objects:
        negative_samples = random.sample(list(absent_objects), min(num_negative, len(absent_objects)))
        for obj in negative_samples:
            question = random.choice(presence_templates).format(obj)
            qas.append({"question": question, "answer": "no"})
            
    return qas

def create_final_qa_dataset(simplified_labels_path, analysis_path, output_path):
    print(f"{Colors.HEADER}--- Starting Final High-Quality QA Dataset Generation ---{Colors.ENDC}")
    
    with open(simplified_labels_path, 'r') as f:
        simplified_data = json.load(f)
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)
        
    all_objects = set(analysis_data['object_category_distribution']['category_counts'].keys())
    
    final_flat_qas = []
    
    # Combine train and val images into one list to process
    all_images = simplified_data.get('bdd100k_labels_images_train.json', []) + \
                 simplified_data.get('bdd100k_labels_images_val.json', [])

    print(f"Found {len(all_images)} total images to process.")

    for image_info in tqdm(all_images, desc="Generating balanced QAs"):
        qas = generate_balanced_qas_for_image(image_info, all_objects)
        if qas:
            final_flat_qas.append({
                "image": image_info['name'],
                "qas": qas
            })
            
    with open(output_path, 'w') as f:
        json.dump(final_flat_qas, f, indent=4)
        
    print(f"\n{Colors.GREEN}--- Generation Complete ---{Colors.ENDC}")
    print(f"Saved {len(final_flat_qas)} image entries to '{output_path}'")
    print("This file is now the master source for QA data and is ready to be split.")

if __name__ == '__main__':
    # Input files
    SIMPLIFIED_LABELS_FILE = 'data/simplified_labels_for_vlm.json'
    ANALYSIS_FILE = 'data/final_dataset_analysis.json'
    
    # The final output file that will be used by split_dataset.py
    FINAL_QA_MASTER_FILE = 'data/final_qa_master_high_quality.json'
    
    create_final_qa_dataset(SIMPLIFIED_LABELS_FILE, ANALYSIS_FILE, FINAL_QA_MASTER_FILE)