import json
from collections import defaultdict
import os
import os
import torch
import random
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel
import os
import json
import shutil
from tqdm import tqdm


class Data():
    def __init__(self):
        pass

    def load_json_files_to_dict(self, file_path1, file_path2):
        """
        Loads two JSON files from their paths and stores their contents in a
        dictionary with the file names as keys.

        Args:
            file_path1 (str): The path to the first JSON file.
            file_path2 (str): The path to the second JSON file.

        Returns:
            dict: A dictionary containing the data from the two JSON files.
                  Returns an empty dictionary if any errors occur.
        """
        combined_data = {}

        for file_path in [file_path1, file_path2]:
            try:
                # Extract the file name from the path to use as a key
                file_name = os.path.basename(file_path)

                with open(file_path, 'r') as file:
                    # Load the JSON data and store it in the dictionary
                    combined_data[file_name] = json.load(file)

            except FileNotFoundError:
                print(f"Error: The file at {file_path} was not found.")
                return {}
            except json.JSONDecodeError:
                print(f"Error: The file at {file_path} is not a valid JSON file.")
                return {}
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return {}

        return combined_data

    # def save_dict_to_json(self, data_dictionary, output_file_path):
    #     """
    #     Saves a dictionary to a file in JSON format.
    #
    #     Args:
    #         data_dictionary (dict): The dictionary to be saved.
    #         output_file_path (str): The path of the file to save the JSON data to.
    #
    #     Returns:
    #         bool: True if the file was saved successfully, False otherwise.
    #     """
    #     try:
    #         with open(output_file_path, 'w') as json_file:
    #             # Use json.dump() to write the dictionary to the file
    #             # The 'indent' parameter is used to format the JSON for readability
    #             json.dump(data_dictionary, json_file, indent=4)
    #         print(f"Successfully saved data to {output_file_path}")
    #         return True
    #     except TypeError as e:
    #         print(f"Error: An object in the dictionary is not JSON serializable. {e}")
    #         return False
    #     except IOError as e:
    #         print(f"Error: Could not write to the file at {output_file_path}. {e}")
    #         return False
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return False



    def get_type_name(self, value):
        """Helper function to get the string name of a variable's type."""
        return type(value).__name__

    def analyze_json_dataset(self, input_file_path):
        """
        Analyzes a merged JSON dataset file to extract metadata about its contents.

        Args:
            input_file_path (str): The path to the merged JSON file.

        Returns:
            dict: A dictionary containing the analysis results.
                  Returns an empty dictionary if the file cannot be processed.
        """
        try:
            with open(input_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file at {input_file_path} was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file at {input_file_path}.")
            return {}

        analysis_results = {}
        all_unique_categories = set()

        # Process each part of the dataset (e.g., train and val)
        for key, image_list in data.items():
            dataset_type = 'train' if 'train' in key else 'val'

            image_count = len(image_list)
            category_counts = defaultdict(int)
            category_properties = {}

            # Iterate through each image and its labels
            for image in image_list:
                if 'labels' in image and image['labels'] is not None:
                    for label in image['labels']:
                        category = label.get('category')
                        if category:
                            # Increment the count for this category
                            category_counts[category] += 1
                            all_unique_categories.add(category)

                            # If we haven't analyzed this category's properties yet, do so now
                            if category not in category_properties:
                                properties = {}
                                for prop, value in label.items():
                                    if isinstance(value, dict):
                                        # For nested dictionaries, get the types of their keys
                                        properties[prop] = {k: self.get_type_name(v) for k, v in value.items()}
                                    else:
                                        properties[prop] = self.get_type_name(value)
                                category_properties[category] = properties

            # Store the results for this dataset type
            analysis_results[dataset_type] = {
                'image_count': image_count,
                'category_counts': dict(category_counts),
                'category_properties': category_properties
            }

        # Add a summary section
        analysis_results['summary'] = {
            'total_images': sum(v['image_count'] for v in analysis_results.values()),
            'unique_categories_across_all_sets': sorted(list(all_unique_categories))
        }

        return {'dataset_analysis': analysis_results}

    def save_dict_to_json(self, data_dictionary, output_file_path):
        """
        Saves a dictionary to a file in JSON format.
        """
        try:
            with open(output_file_path, 'w') as json_file:
                json.dump(data_dictionary, json_file, indent=4)
            print(f"Successfully saved analysis to {output_file_path}")
            return True
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")
            return False

    import json

    def simplify_dataset_for_vlm(self, input_file_path, output_file_path):
        """
        Simplifies a merged JSON dataset to retain only the information needed for
        building a simple Q&A dataset for a VLM.

        This function extracts:
        - The weather for each image.
        - A list of unique object categories in each image.
        - The color of any traffic lights present.

        Args:
            input_file_path (str): The path to the large, merged JSON file.
            output_file_path (str): The path where the simplified JSON will be saved.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with open(input_file_path, 'r') as f:
                original_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{input_file_path}' was not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{input_file_path}'.")
            return False

        simplified_data = {}

        # Process both 'train' and 'val' sections of the data
        for key, image_list in original_data.items():
            simplified_image_list = []

            for image in image_list:
                # Extract basic image information
                image_name = image.get('name')
                weather = image.get('attributes', {}).get('weather')

                unique_categories = set()
                traffic_lights = []

                # Process labels if they exist
                if 'labels' in image and image['labels'] is not None:
                    for label in image['labels']:
                        category = label.get('category')
                        if category:
                            unique_categories.add(category)

                            # Specifically check for traffic lights and their color
                            if category == 'traffic light':
                                color = label.get('attributes', {}).get('trafficLightColor')
                                if color and color != 'none':
                                    traffic_lights.append({'color': color})

                # Create the new, simplified structure for this image
                simplified_image_info = {
                    'name': image_name,
                    'weather': weather,
                    'objects': sorted(list(unique_categories)),  # Convert set to a sorted list
                    'traffic_lights': traffic_lights
                }
                simplified_image_list.append(simplified_image_info)

            simplified_data[key] = simplified_image_list

        # Save the simplified data to a new JSON file
        try:
            with open(output_file_path, 'w') as f:
                json.dump(simplified_data, f, indent=4)
            print(f"Successfully created simplified JSON file at: {output_file_path}")
            return True
        except IOError as e:
            print(f"Error writing to file: {e}")
            return False

    def find_all_images_in_directory(self, directory):
        """
        Recursively finds all image files (.jpg, .jpeg, .png) in a directory
        and its subdirectories.

        Args:
            directory (str): The path to the root directory to search.

        Returns:
            set: A set containing the base names of all found image files.
        """
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_filenames = set()

        print(f"Scanning directory '{directory}' for images...")

        if not os.path.isdir(directory):
            print(f"Error: Directory not found at '{directory}'")
            return image_filenames

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_filenames.add(file)

        print(f"Found {len(image_filenames)} unique image files on disk.")
        return image_filenames

    def filter_and_flatten_labels(self, image_directory, simplified_labels_path, output_path, max_images=10000):
        """
        Filters labels based on available images on disk, flattens the data structure,
        and saves a specified number of labeled entries to a new JSON file.

        Args:
            image_directory (str): Path to the directory containing all image files.
            simplified_labels_path (str): Path to the 'simplified_labels_for_vlm.json' file.
            output_path (str): Path to save the final, filtered, and flattened JSON file.
            max_images (int): The maximum number of image labels to include in the final file.

        Returns:
            bool: True if successful, False otherwise.
        """
        # Step 1: Get the set of all available images from the directory
        available_images = self.find_all_images_in_directory(image_directory)
        if not available_images:
            print("No images found. Aborting.")
            return False

        # Step 2: Load the simplified labels JSON file
        try:
            with open(simplified_labels_path, 'r') as f:
                simplified_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: The label file '{simplified_labels_path}' was not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{simplified_labels_path}'.")
            return False

        # Step 3: Filter labels and flatten the structure
        final_labels_list = []
        print(f"Filtering labels against available images. Aiming for {max_images} entries.")

        # Iterate through the lists under keys like "bdd100k_labels_images_train.json", etc.
        for key in simplified_data:
            image_list = simplified_data[key]
            for label_info in image_list:
                # Check if we have reached our target number of images
                if len(final_labels_list) >= max_images:
                    break

                # Check if the image for this label exists on disk
                if label_info.get('name') in available_images:
                    final_labels_list.append(label_info)

            if len(final_labels_list) >= max_images:
                print("Reached the maximum desired number of images.")
                break

        print(f"Created a final list with {len(final_labels_list)} labeled images.")

        # Step 4: Save the new, flattened JSON file
        try:
            with open(output_path, 'w') as f:
                json.dump(final_labels_list, f, indent=4)
            print(f"Successfully saved the final dataset to '{output_path}'")
            return True
        except IOError as e:
            print(f"Error: Could not write to the output file at '{output_path}'. {e}")
            return False

    def analyze_final_dataset(self, input_file_path):
        """
        Analyzes the final, flattened VLM dataset to provide a statistical overview.

        Args:
            input_file_path (str): The path to the 'final_vlm_dataset_10k.json' file.

        Returns:
            dict: A dictionary containing the detailed analysis of the dataset.
        """
        try:
            with open(input_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{input_file_path}' was not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{input_file_path}'.")
            return None

        if not data:
            print("Warning: The dataset is empty.")
            return {
                "error": "The provided JSON file is empty or invalid."
            }

        # --- Initialize counters and structures for analysis ---
        total_images = len(data)
        weather_counts = defaultdict(int)
        object_category_counts = defaultdict(int)
        traffic_light_color_counts = defaultdict(int)

        # --- Analyze the data structure from the first entry ---
        first_entry = data[0]
        data_structure = {}
        for key, value in first_entry.items():
            if isinstance(value, list) and value:
                # Handle list type, checking the type of its elements
                element = value[0]
                if isinstance(element, dict):
                    # For a list of dictionaries, like 'traffic_lights'
                    nested_structure = {k: self.get_type_name(v) for k, v in element.items()}
                    data_structure[key] = f"list of dicts with structure: {nested_structure}"
                else:
                    # For a list of simple types, like 'objects'
                    data_structure[key] = f"list of {self.get_type_name(element)}"
            else:
                data_structure[key] = self.get_type_name(value)

        # --- Iterate through the entire dataset to aggregate statistics ---
        for item in data:
            # Count weather conditions
            if 'weather' in item:
                weather_counts[item['weather']] += 1

            # Count each object category
            if 'objects' in item and item['objects']:
                for obj in item['objects']:
                    object_category_counts[obj] += 1

            # Count traffic light colors
            if 'traffic_lights' in item and item['traffic_lights']:
                for light in item['traffic_lights']:
                    if 'color' in light:
                        traffic_light_color_counts[light['color']] += 1

        # --- Assemble the final analysis report ---
        analysis_report = {
            "dataset_summary": {
                "total_images_analyzed": total_images,
                "data_structure_and_types": data_structure
            },
            "attribute_distributions": {
                "weather_conditions": dict(sorted(weather_counts.items(), key=lambda item: item[1], reverse=True)),
                "traffic_light_colors": dict(
                    sorted(traffic_light_color_counts.items(), key=lambda item: item[1], reverse=True))
            },
            "object_category_distribution": {
                "total_unique_categories": len(object_category_counts),
                "category_counts": dict(sorted(object_category_counts.items(), key=lambda item: item[1], reverse=True))
            }
        }

        return analysis_report

    def save_analysis_to_json(self, analysis_data, output_path):
        """Saves the analysis dictionary to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis_data, f, indent=4)
            print(f"Successfully saved analysis report to '{output_path}'")
            return True
        except IOError as e:
            print(f"Error: Could not write to the output file '{output_path}'. {e}")
            return False

    def build_image_path_map(self, root_dir):
        """
        Recursively scans a directory and creates a map of image filenames to their full paths.
        This is much faster than searching for each file individually in a loop.
        """
        print(f"Building a map of all images in '{root_dir}'... This may take a moment.")
        path_map = {}
        image_extensions = ('.jpg', '.jpeg', '.png')
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(image_extensions):
                    if filename not in path_map:
                        path_map[filename] = os.path.join(dirpath, filename)
        print(f"Found {len(path_map)} unique images in the source directory.")
        return path_map

    def generate_qas_for_image(self, image_info, all_unique_categories):
        """Generates a list of question-answer pairs for a single image's data."""
        qas = []

        # Extract details from the label
        objects = image_info.get("objects", [])
        traffic_lights = image_info.get("traffic_lights", [])

        # 1. QA for all objects
        qas.append({
            "question": "What are the objects in the image?",
            "answer": ", ".join(sorted(objects))
        })

        # 2. QA for traffic light presence
        qas.append({
            "question": "Is there a traffic light in the image?",
            "answer": "yes" if "traffic light" in objects else "no"
        })

        # 3. QA for the presence of each unique object type
        for category in all_unique_categories:
            # Simple grammar for 'a' vs 'an'
            article = 'an' if category.startswith(('a', 'e', 'i', 'o', 'u')) else 'a'
            qas.append({
                "question": f"Is there {article} {category} in the image?",
                "answer": "yes" if category in objects else "no"
            })

        # 4. QA for the color of the traffic light
        if traffic_lights:
            # For simplicity, we'll use the color of the first traffic light listed
            first_light_color = traffic_lights[0].get("color", "unknown")
            qas.append({
                "question": "What is the color of the traffic light?",
                "answer": first_light_color
            })

        return qas

    def prepare_final_vlm_dataset(self, raw_image_root_dir, final_labels_path, analysis_path, output_dir):
        """
        Creates the final processed dataset directory with hard copies of images and a Q&A JSON file.
        """
        print("--- Starting Final VLM Dataset Preparation ---")

        # 1. Setup output directories
        processed_images_dir = os.path.join(output_dir, 'images')
        os.makedirs(processed_images_dir, exist_ok=True)
        print(f"Output directory created at: '{output_dir}'")

        # 2. Load the label and analysis files
        try:
            with open(final_labels_path, 'r') as f:
                labels_data = json.load(f)
            with open(analysis_path, 'r') as f:
                analysis_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: Could not find a required file. {e}")
            return

        # Extract the master list of all unique categories from the analysis file
        unique_categories = list(analysis_data['object_category_distribution']['category_counts'].keys())

        # 3. Build a map of all source images for fast lookups
        source_image_map = self.build_image_path_map(raw_image_root_dir)
        if not source_image_map:
            print("Error: No source images were found. Please check the `RAW_IMAGE_ROOT_DIR` path.")
            return

        # 4. Process each image: copy it and generate its Q&As
        final_qas = []

        for image_info in tqdm(labels_data, desc="Processing dataset"):
            image_name = image_info.get("name")

            # Find the source path from our map
            source_path = source_image_map.get(image_name)

            if source_path:
                # Copy the image to the new processed directory
                dest_path = os.path.join(processed_images_dir, image_name)
                shutil.copy2(source_path, dest_path)

                # Generate the Q&As for this image
                qas = self.generate_qas_for_image(image_info, unique_categories)

                # Add to our final list, linking the Q&As to the image file name
                final_qas.append({
                    "image": image_name,
                    "qas": qas
                })
            else:
                print(f"Warning: Could not find image '{image_name}' in the source directory. Skipping.")

        # 5. Save the final Q&A JSON file
        qa_output_path = os.path.join(output_dir, 'questions_and_answers.json')
        with open(qa_output_path, 'w') as f:
            json.dump(final_qas, f, indent=4)

        print("\n--- Dataset Preparation Complete! ---")
        print(f"Total images copied: {len(final_qas)}")
        print(f"Final Q&A file saved to: '{qa_output_path}'")

    def create_and_save_embeddings(self, image_dir, output_dir, model_name='openai/clip-vit-base-patch32'):
        """
        Generates and saves image embeddings using a pre-trained vision model.
        """
        print("--- Starting Image Embedding Creation ---")

        # 1. Setup device and output directory
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Using device: {device}")
        print(f"Embeddings will be saved to: '{output_dir}'")

        # 2. Load the pre-trained model and processor
        try:
            model = CLIPVisionModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
            model.eval()  # Set model to evaluation mode
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have an internet connection and the 'transformers' library is installed.")
            return

        # 3. Get the list of images to process
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 4. Process each image and save its embedding
        print(f"Found {len(image_files)} images to process.")

        with torch.no_grad():  # Disable gradient calculations for inference
            for image_name in tqdm(image_files, desc="Generating Embeddings"):
                image_path = os.path.join(image_dir, image_name)

                # Define the output path for the embedding
                base_name = os.path.splitext(image_name)[0]
                output_path = os.path.join(output_dir, f"{base_name}.pt")

                # Skip if embedding already exists
                if os.path.exists(output_path):
                    continue

                try:
                    # Open image and preprocess
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)

                    # Get the embedding
                    # The 'pooler_output' gives a single vector representation for the entire image
                    outputs = model(**inputs)
                    embedding = outputs.pooler_output.squeeze().cpu()  # Move to CPU for saving

                    # Save the embedding tensor
                    torch.save(embedding, output_path)

                except Exception as e:
                    print(f"\nCould not process image {image_name}. Error: {e}")

        print("\n--- Embedding Creation Complete! ---")
        print(f"All embeddings have been saved in '{output_dir}'")

    def analyze_single_embedding(self, embeddings_dir, output_json_path):
        """
        Analyzes a single sample embedding to determine its properties and saves
        the analysis to a JSON file.

        Args:
            embeddings_dir (str): The directory containing the .pt embedding files.
            output_json_path (str): The path to save the analysis JSON report.
        """
        print("--- Starting Embedding Analysis ---")

        # 1. Find a sample embedding file to analyze
        try:
            sample_file_name = next(f for f in os.listdir(embeddings_dir) if f.endswith('.pt'))
            sample_file_path = os.path.join(embeddings_dir, sample_file_name)
            print(f"Found sample embedding file to analyze: '{sample_file_name}'")
        except StopIteration:
            print(f"Error: No .pt files found in the directory '{embeddings_dir}'.")
            print("Please run the 'create_embeddings.py' script first.")
            return
        except FileNotFoundError:
            print(f"Error: The directory '{embeddings_dir}' does not exist.")
            return

        # 2. Load the embedding tensor
        try:
            embedding_tensor = torch.load(sample_file_path)
            print("Embedding tensor loaded successfully.")
        except Exception as e:
            print(f"Error loading the tensor from '{sample_file_path}': {e}")
            return

        # 3. Extract all relevant properties
        analysis = {
            "sample_file": sample_file_name,
            "properties": {
                "data_type": str(embedding_tensor.dtype),
                "shape": list(embedding_tensor.shape),
                "dimensions": embedding_tensor.dim(),
                "total_elements": embedding_tensor.numel(),
                "device": str(embedding_tensor.device),
                "memory_usage_bytes": embedding_tensor.element_size() * embedding_tensor.numel(),
                "value_range": {
                    "min": float(embedding_tensor.min().item()),
                    "max": float(embedding_tensor.max().item()),
                    "mean": float(embedding_tensor.mean().item())
                }
            },
            "implications_for_vlm_architecture": {
                "projector_input_dimension": embedding_tensor.shape[0] if embedding_tensor.dim() > 0 else 0,
                "notes": "This analysis provides the blueprint for the input layer of the projector module."
            }
        }

        # 4. Save the analysis to a JSON file
        try:
            with open(output_json_path, 'w') as f:
                json.dump(analysis, f, indent=4)
            print(f"\n--- Analysis Complete ---")
            print(f"Report saved to: '{output_json_path}'")

            # Also print the key findings to the console
            print("\nKey Findings:")
            print(
                f"  - Embedding Dimension: {analysis['implications_for_vlm_architecture']['projector_input_dimension']}")
            print(f"  - Data Type: {analysis['properties']['data_type']}")

        except IOError as e:
            print(f"Error saving the analysis file: {e}")

    def split_dataset(self, base_dir, output_dir, split_ratios=(0.8, 0.1, 0.1), random_seed=42):
        """
        Splits the embeddings and QA data into train, validation, and test sets.

        Args:
            base_dir (str): The path to the 'processed_vlm_dataset' directory.
            output_dir (str): The path to the new directory for the split dataset (e.g., 'final_dataset').
            split_ratios (tuple): A tuple with the ratios for (train, val, test).
            random_seed (int): A seed for the random shuffle to ensure reproducibility.
        """
        print("--- Starting Dataset Split ---")

        # 1. Define source paths
        source_embeddings_dir = os.path.join(base_dir, 'embeddings')
        source_qa_path = os.path.join(base_dir, 'questions_and_answers.json')

        # 2. Load the main QA JSON file, which serves as our master list
        try:
            with open(source_qa_path, 'r') as f:
                qa_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: QA file not found at '{source_qa_path}'")
            return

        # 3. Shuffle the data for a random split
        random.seed(random_seed)
        random.shuffle(qa_data)
        print(f"Loaded and shuffled {len(qa_data)} total samples using random seed {random_seed}.")

        # 4. Calculate split points
        total_samples = len(qa_data)
        train_end = int(total_samples * split_ratios[0])
        val_end = train_end + int(total_samples * split_ratios[1])

        # 5. Create the split lists
        train_data = qa_data[:train_end]
        val_data = qa_data[train_end:val_end]
        test_data = qa_data[val_end:]

        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        # 6. Create directories and process each split
        os.makedirs(output_dir, exist_ok=True)

        for split_name, split_data in splits.items():
            print(f"\nProcessing '{split_name}' split with {len(split_data)} samples...")

            # Create subdirectories for this split
            split_output_dir = os.path.join(output_dir, split_name)
            split_embeddings_dir = os.path.join(split_output_dir, 'embeddings')
            os.makedirs(split_embeddings_dir, exist_ok=True)

            # Save the QA JSON for this split
            split_qa_path = os.path.join(split_output_dir, 'qa.json')
            with open(split_qa_path, 'w') as f:
                json.dump(split_data, f, indent=4)

            # Copy the corresponding embedding files
            for item in tqdm(split_data, desc=f"Copying {split_name} embeddings"):
                image_name = item['image']
                base_name = os.path.splitext(image_name)[0]
                embedding_filename = f"{base_name}.pt"

                source_path = os.path.join(source_embeddings_dir, embedding_filename)
                dest_path = os.path.join(split_embeddings_dir, embedding_filename)

                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                else:
                    print(f"Warning: Embedding file not found for '{image_name}'. Skipping.")

        print("\n--- Dataset Split Complete! ---")
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        print(f"Final dataset created at: '{output_dir}'")


if __name__ == "__main__":
    
    data = Data()

    ################## merge the labels jsons ##################
    train_label_path = "data/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    valid_label_path = "data/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    labels_dict = data.load_json_files_to_dict(train_label_path, valid_label_path)
    
    print(type(labels_dict))
    print(len(labels_dict))
    
    data.save_dict_to_json(labels_dict, "data/merged_labels.json")
    
    #################### prune the labels based in the target labels we want ########################
    # Path to your original merged JSON file
    input_path = 'data/merged_labels.json'  # Make sure this is the correct name
    
    # Path for the new, simplified JSON file
    output_path = 'data/simplified_labels_for_vlm.json'
    
    # Run the simplification process
    data.simplify_dataset_for_vlm(input_path, output_path)
