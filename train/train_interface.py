import yaml
import json
import datetime
import torch
import os
from train.trainer import VLMTrainer, VQADataset

class TrainingInterface:
    def __init__(self, config_path='config.yml'):
        print("--- Initializing Training Interface ---")
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._get_device()

    def _load_config(self):
        print(f"Loading configuration from: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_device(self):
        if torch.backends.mps.is_available(): return "mps"
        elif torch.cuda.is_available(): return "cuda"
        else: return "cpu"

    def _generate_training_plan(self):
        print("Generating detailed training plan...")
        plan = {}
        
        # 1. Add config and execution details
        plan['training_parameters'] = self.config
        plan['execution_details'] = {
            'timestamp_utc': datetime.datetime.utcnow().isoformat(),
            'device': self.device
        }

        # 2. Add dataset details
        # We need to instantiate the datasets to get their lengths and samples
        temp_train_dataset = VQADataset(
            qa_path=os.path.join(self.config['dataset_path'], 'train/qa.json'),
            embeddings_dir=os.path.join(self.config['dataset_path'], 'train/embeddings'),
            max_samples=self.config['sampling']['max_train_samples']
        )
        temp_val_dataset = VQADataset(
            qa_path=os.path.join(self.config['dataset_path'], 'val/qa.json'),
            embeddings_dir=os.path.join(self.config['dataset_path'], 'val/embeddings'),
            max_samples=self.config['sampling']['max_val_samples']
        )
        
        total_train_qas = len(temp_train_dataset)
        total_val_qas = len(temp_val_dataset)
        
        plan['dataset_summary'] = {
            'total_train_qa_pairs': total_train_qas,
            'total_val_qa_pairs': total_val_qas,
            'train_batch_size': self.config['training']['batch_size'],
            'val_batch_size': self.config['training']['batch_size'],
            'batches_per_train_epoch': (total_train_qas + self.config['training']['batch_size'] - 1) // self.config['training']['batch_size']
        }
        
        # 3. Show a sample of the data to be used
        plan['data_samples'] = {
            'note': 'Showing the first few QA pairs from the training set to verify data structure.',
            'train_samples': []
        }
        num_samples_to_show = 5
        for i in range(min(num_samples_to_show, total_train_qas)):
             _, q, a = temp_train_dataset[i]
             plan['data_samples']['train_samples'].append({'question': q, 'answer': a})
             
        # 4. Describe the training logic
        plan['training_logic'] = {
            'total_epochs': self.config['training']['epochs'],
            'optimizer': 'Adam',
            'learning_rate': self.config['training']['learning_rate'],
            'loss_calculation_note': f"Loss is calculated only on answer tokens because mask_prompt_labels is set to {self.config['training']['mask_prompt_labels']}. This is critical for stability.",
            'checkpointing': f"Model checkpoints (projector and LoRA adapters) are saved to '{self.config['checkpoint_dir']}' whenever validation loss improves."
        }
        
        return plan

    def run(self):
        # Generate and save the plan *before* starting the training
        training_plan = self._generate_training_plan()
        plan_path = 'training_plan.json'
        with open(plan_path, 'w') as f:
            json.dump(training_plan, f, indent=4)
        print(f"Training plan saved to '{plan_path}'. You can inspect this file now.")
        print("\n--- Starting Training Process ---")
        
        # Instantiate and run the trainer
        trainer = VLMTrainer(self.config)
        trainer.train()
        print("\n--- Training Finished ---")

if __name__ == '__main__':
    interface = TrainingInterface(config_path='config.yml')
    interface.run()