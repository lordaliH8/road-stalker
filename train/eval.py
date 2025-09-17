import os
import json
import torch
import yaml
from tqdm import tqdm
from collections import defaultdict
import evaluate
import nltk

# --- CRITICAL: Import our custom classes from the trainer module ---
from train.trainer import VLM, QuantumProjector

# --- ANSI color codes ---
class Colors:
    GREEN = '\033[92m'; BLUE = '\033[94m'; HEADER = '\033[95m'; ENDC = '\033[0m'; YELLOW = '\033[93m'; RED = '\033[91m'

# --- NLTK Downloader ---
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print(f"{Colors.YELLOW}Downloading NLTK data packages for METEOR score...{Colors.ENDC}")
    nltk.download('wordnet')
    nltk.download('punkt')

# --- Helper Functions ---
def categorize_question(question_text):
    q_lower = question_text.lower()
    if "weather" in q_lower: return "Weather"
    elif "color" in q_lower: return "Traffic Light Color"
    elif "visible" in q_lower or "contain" in q_lower: return "Object Presence"
    elif "objects" in q_lower: return "Objects Listing"
    else: return "Other"

def are_answers_correct_simple(prediction, ground_truth):
    return prediction.lower().strip() == ground_truth.lower().strip()

# --- Main Evaluation Class ---
class VQAEvaluator:
    def __init__(self, config_path='config.yml', checkpoint_dir='checkpoints/qvlm_checkpoints'):
        print(f"{Colors.HEADER}======================================={Colors.ENDC}")
        print(f"{Colors.HEADER}      Running Full Model Evaluation      {Colors.ENDC}")
        print(f"{Colors.HEADER}======================================={Colors.ENDC}")

        with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        
        self._initialize_inference_engine(checkpoint_dir)
        self._load_test_data()

    def _initialize_inference_engine(self, checkpoint_dir):
        # This mirrors the robust loading logic from our final inference script
        print(f"\n{Colors.BLUE}--- Initializing Inference Engine... ---{Colors.ENDC}")
        # We create a temporary instance of our inference class to reuse its loading logic
        from train.inference import VLMInference
        self.inference_engine = VLMInference(config_path='config.yml', checkpoint_dir=checkpoint_dir)

    def _load_test_data(self):
        test_qa_path = os.path.join(self.config['dataset_path'], 'test/qa.json')
        self.test_embeddings_dir = os.path.join(self.config['dataset_path'], 'test/embeddings')
        with open(test_qa_path, 'r') as f: test_data = json.load(f)
        # Flatten the data for easier iteration
        self.flat_test_data = [(item['image'], qa) for item in test_data for qa in item['qas']]
        print(f"Loaded {len(self.flat_test_data)} QA pairs from the test set.")

    def run(self):
        print(f"\n{Colors.BLUE}--- Running inference on all test set QA pairs... ---{Colors.ENDC}")
        results = []
        for image_name, qa_pair in tqdm(self.flat_test_data, desc="Evaluating Test Set"):
            embedding_path = os.path.join(self.test_embeddings_dir, os.path.splitext(image_name)[0] + '.pt')
            question, ground_truth = qa_pair['question'], qa_pair['answer']
            
            # Use the inference engine to get the prediction
            prediction = self.inference_engine.answer_question(embedding_path, question)
            
            results.append({
                "image": image_name,
                "category": categorize_question(question),
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction
            })

        self._calculate_and_display_metrics(results)

    def _calculate_and_display_metrics(self, results):
        print(f"\n{Colors.BLUE}--- Calculating All Metrics... ---{Colors.ENDC}")
        metrics = {}
        
        # 1. Accuracy for Simple Categories
        simple_categories = ["Weather", "Traffic Light Color", "Object Presence"]
        category_stats = {cat: {'correct': 0, 'total': 0} for cat in simple_categories}
        incorrect_simple_predictions = []
        for r in results:
            if r['category'] in simple_categories:
                cat = r['category']
                is_correct = are_answers_correct_simple(r['prediction'], r['ground_truth'])
                category_stats[cat]['total'] += 1
                if is_correct: category_stats[cat]['correct'] += 1
                else: incorrect_simple_predictions.append(r)
        total_simple_correct = sum(s['correct'] for s in category_stats.values())
        total_simple = sum(s['total'] for s in category_stats.values())
        metrics['overall_accuracy_simple_qa'] = total_simple_correct / total_simple if total_simple > 0 else 0
        metrics['per_category_accuracy'] = {cat: s['correct'] / s['total'] if s['total'] > 0 else 0 for cat, s in category_stats.items()}

        # 2. Confusion Matrix & Advanced Metrics for "Object Presence"
        tp, tn, fp, fn = 0, 0, 0, 0
        for r in [res for res in results if res['category'] == 'Object Presence']:
            is_yes_truth = r['ground_truth'].lower() == 'yes'; is_yes_pred = r['prediction'].lower() == 'yes'
            if is_yes_pred and is_yes_truth: tp += 1
            elif not is_yes_pred and not is_yes_truth: tn += 1
            elif is_yes_pred and not is_yes_truth: fp += 1
            elif not is_yes_pred and is_yes_truth: fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0; recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics['object_presence_metrics'] = {"confusion_matrix": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}, "precision": precision, "recall": recall, "f1_score": f1_score}
        
        # 3. Generative Metrics for "Objects Listing"
        generative_results = [r for r in results if r['category'] == 'Objects Listing']
        if generative_results:
            predictions = [r['prediction'] for r in generative_results]
            references = [[r['ground_truth']] for r in generative_results]
            bleu = evaluate.load('bleu'); rouge = evaluate.load('rouge'); meteor = evaluate.load('meteor')
            metrics['objects_listing_metrics'] = {
                "bleu": bleu.compute(predictions=predictions, references=references)['bleu'],
                "rougeL": rouge.compute(predictions=predictions, references=references)['rougeL'],
                "meteor": meteor.compute(predictions=predictions, references=references)['meteor']
            }
        
        # 4. Print and Save Final Report
        print(f"\n{Colors.HEADER}======================================={Colors.ENDC}")
        print(f"{Colors.HEADER}          Evaluation Report            {Colors.ENDC}")
        print(f"{Colors.HEADER}======================================={Colors.ENDC}")
        print(f"  {Colors.BLUE}Overall Accuracy (Simple Questions):{Colors.ENDC} {metrics['overall_accuracy_simple_qa']:.2%}")
        print(f"\n  {Colors.BLUE}Accuracy by Question Type:{Colors.ENDC}")
        for cat, acc in metrics['per_category_accuracy'].items():
            print(f"    - {cat:<20}: {acc:.2%}")
        print(f"\n  {Colors.BLUE}Detailed Metrics for 'Object Presence' (Yes/No):{Colors.ENDC}")
        print(f"    - Precision: {metrics['object_presence_metrics']['precision']:.2%}  (When it says 'yes', how often is it right?)")
        print(f"    - Recall:    {metrics['object_presence_metrics']['recall']:.2%}  (Of all true 'yes' cases, how many did it find?)")
        print(f"    - F1-Score:  {metrics['object_presence_metrics']['f1_score']:.2%}  (Harmonic mean of Precision and Recall)")
        print(f"    - Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        if 'objects_listing_metrics' in metrics:
            print(f"\n  {Colors.BLUE}Metrics for 'Objects Listing' (Generative Task):{Colors.ENDC}")
            print(f"    - BLEU:      {metrics['objects_listing_metrics']['bleu']:.4f} (Measures n-gram precision)")
            print(f"    - ROUGE-L:   {metrics['objects_listing_metrics']['rougeL']:.4f} (Measures recall of the longest common sequence)")
            print(f"    - METEOR:    {metrics['objects_listing_metrics']['meteor']:.4f} (Harmonic mean of precision/recall with synonym matching)")
        
        output_report = {"metrics": metrics, "incorrect_simple_predictions": incorrect_simple_predictions, "generative_predictions_and_references": generative_results}
        output_path = "evaluation_results_quantum.json"
        with open(output_path, 'w') as f: json.dump(output_report, f, indent=4)
        print(f"\n{Colors.GREEN}--- Full report with incorrect predictions saved to '{output_path}' ---{Colors.ENDC}")

if __name__ == '__main__':
    # Make sure to have a config.yml and a trained checkpoint available
    # The script will use the paths defined in your config.yml
    evaluator = VQAEvaluator()
    evaluator.run()