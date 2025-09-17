import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import math

# --- QML Imports ---
import pennylane as qml
from pennylane import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# --- ANSI color codes ---
class Colors:
    GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'; BLUE = '\033[94m'; HEADER = '\033[95m'; ENDC = '\033[0m'



# --- Quantum Device Setup ---
# This is defined globally as it's a static resource for the simulation.
NUM_QUBITS = 16
dev = qml.device("default.qubit", wires=NUM_QUBITS)

# --- Quantum Projector Definition (MODIFIED) ---
class QuantumProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_qubits):
        super().__init__()
        self.pre_net = nn.Linear(input_dim, num_qubits)
        self.post_net = nn.Linear(num_qubits, output_dim)
        
        # --- THE FIX IS HERE: The interface MUST be 'torch' for TorchLayer ---
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits))
            for i in range(num_qubits): qml.RY(weights[i], wires=i)
            for i in range(num_qubits - 1): qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        weight_shapes = {"weights": num_qubits}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        # x is a torch.Tensor on the primary device (e.g., MPS)
        
        # 1. Classical pre-processing
        x = self.pre_net(x)
        
        # 2. CPU Isolation for the quantum part
        # Pull to CPU before quantum layer, as its backend is CPU-based
        x_cpu = x.cpu()
        x_q_cpu = self.q_layer(x_cpu)
        
        # 3. Push result back to the original device and dtype
        x_q_gpu = x_q_cpu.to(x.device).to(x.dtype)
        
        # 4. Classical post-processing
        x = self.post_net(x_q_gpu)
        return x



# --- Core VLM Model Class ---
class VLM(nn.Module):
    # THE FIX: VLM's constructor now accepts the device.
    def __init__(self, config, device):
        super().__init__()
        
        if config.get('qml') and config['qml'].get('enabled'):
            print(f"{Colors.BLUE}--- Using Quantum Projector ---{Colors.ENDC}")
            self.projector = QuantumProjector(
                input_dim=config['model']['vision_embedding_dim'],
                output_dim=config['model']['llm_embedding_dim'],
                num_qubits=config['qml']['num_qubits']
            )
        else:
            print(f"{Colors.BLUE}--- Using Classical Projector ---{Colors.ENDC}")
            self.projector = nn.Sequential(
                nn.Linear(config['model']['vision_embedding_dim'], config['model']['projector_hidden_dim']),
                nn.GELU(),
                nn.Linear(config['model']['projector_hidden_dim'], config['model']['llm_embedding_dim'])
            )
        
        self.llm = AutoModelForCausalLM.from_pretrained(config['model']['llm_name'], dtype=torch.bfloat16)
        self.projector.to(torch.bfloat16)
        
        peft_config = LoraConfig(r=config['lora']['r'], lora_alpha=config['lora']['lora_alpha'], lora_dropout=config['lora']['lora_dropout'], target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
    
    @staticmethod
    def get_tokenizer(llm_name):
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
        return tokenizer

    def forward(self, image_embedding, tokenized_prompt, labels):
        image_embedding = image_embedding.to(self.projector.pre_net.weight.dtype if isinstance(self.projector, QuantumProjector) else self.projector[0].weight.dtype)
        projected_vision_embedding = self.projector(image_embedding)
        text_embeddings = self.llm.get_input_embeddings()(tokenized_prompt['input_ids'])
        inputs_embeds = torch.cat([projected_vision_embedding.unsqueeze(1), text_embeddings[:, 1:, :]], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=tokenized_prompt['attention_mask'], labels=labels)
        return outputs.loss

# --- Main Trainer Class ---
class VLMTrainer:
    def __init__(self, config):
        self.config = config
        if torch.backends.mps.is_available(): self.device = "mps"
        elif torch.cuda.is_available(): self.device = "cuda"
        else: self.device = "cpu"
        
        self.tokenizer = VLM.get_tokenizer(config['model']['llm_name'])
        
        # --- THE FIX: Pass the determined device to the VLM constructor ---
        self.model = VLM(config, self.device).to(self.device)
        self.model.llm.resize_token_embeddings(len(self.tokenizer))
        
        learning_rate = float(config['training']['learning_rate'])
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Use the modern, device-aware GradScaler. Pass device string.
        self.scaler = torch.amp.GradScaler(self.device, enabled=(self.device == 'cuda'))
        
        self._setup_data_loaders()
    
    # ... (The rest of the VLMTrainer and VQADataset class is unchanged from the last valid version) ...
    def _setup_data_loaders(self):
        self.train_dataset = VQADataset(qa_path=os.path.join(self.config['dataset_path'], 'train/qa.json'), embeddings_dir=os.path.join(self.config['dataset_path'], 'train/embeddings'), max_samples=self.config['sampling']['max_train_samples'])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True, collate_fn=self._collate_fn)
        self.val_dataset = VQADataset(qa_path=os.path.join(self.config['dataset_path'], 'val/qa.json'), embeddings_dir=os.path.join(self.config['dataset_path'], 'val/embeddings'), max_samples=self.config['sampling']['max_val_samples'])
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False, collate_fn=self._collate_fn)

    def _collate_fn(self, batch):
        embeddings, questions, answers = zip(*batch)
        prompts_no_answer = [f"<image>\nQuestion: {q}\nAnswer: " for q in questions]
        answers_with_eos = [f" {a}{self.tokenizer.eos_token}" for a in answers]
        tokenized_prompts = self.tokenizer(prompts_no_answer, padding=True, return_tensors='pt')
        tokenized_answers = self.tokenizer(answers_with_eos, padding=True, return_tensors='pt')
        full_input_ids = torch.cat([tokenized_prompts.input_ids, tokenized_answers.input_ids], dim=1)
        full_attention_mask = torch.cat([tokenized_prompts.attention_mask, tokenized_answers.attention_mask], dim=1)
        labels = torch.full_like(full_input_ids, -100)
        prompt_lengths = tokenized_prompts.attention_mask.sum(dim=1)
        for i in range(len(labels)):
            prompt_len = prompt_lengths[i]
            answer_len = tokenized_answers.attention_mask[i].sum()
            labels[i, prompt_len : prompt_len + answer_len] = tokenized_answers.input_ids[i, :answer_len]
        final_tokenized_input = {"input_ids": full_input_ids.to(self.device), "attention_mask": full_attention_mask.to(self.device)}
        return torch.stack(embeddings).to(self.device), final_tokenized_input, labels.to(self.device)

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")
        for embeddings, prompts, labels in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                loss = self.model(embeddings, prompts, labels)
            if math.isnan(loss.item()):
                print("Warning: NaN loss detected. Skipping batch.")
                continue
            if self.device == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            pbar.set_description(f"Training (loss: {loss.item():.4f})")
        return total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0

    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for i, (embeddings, prompts, labels) in enumerate(pbar):
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    loss = self.model(embeddings, prompts, labels)
                if not math.isnan(loss.item()):
                    total_loss += loss.item()
        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
    
    def train(self):
        best_val_loss = float('inf')
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        for epoch in range(self.config['training']['epochs']):
            print(f"\n--- Epoch {epoch+1}/{self.config['training']['epochs']} ---")
            train_loss = self._train_one_epoch()
            val_loss = self._validate()
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            if not math.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"{Colors.GREEN}New best validation loss! Saving model...{Colors.ENDC}")
                projector_path = os.path.join(self.config['checkpoint_dir'], 'best_projector.pth')
                torch.save(self.model.projector.state_dict(), projector_path)
                self.model.llm.save_pretrained(os.path.join(self.config['checkpoint_dir'], 'best_llm_adapters'))
            elif math.isnan(val_loss):
                print(f"{Colors.YELLOW}Validation loss is NaN. Skipping model saving.{Colors.ENDC}")

# Add the VQADataset definition here if it's not in a separate file
class VQADataset(Dataset):
    def __init__(self, qa_path, embeddings_dir, max_samples=None):
        with open(qa_path, 'r') as f: qa_data = json.load(f)
        self.flat_data = []
        for item in qa_data:
            image_name = item['image']
            base_name = os.path.splitext(image_name)[0]
            embedding_path = os.path.join(embeddings_dir, f"{base_name}.pt")
            if os.path.exists(embedding_path):
                for qa_pair in item['qas']: self.flat_data.append({"embedding_path": embedding_path, "question": qa_pair['question'], "answer": qa_pair['answer']})
        if max_samples: self.flat_data = random.sample(self.flat_data, max_samples)

    def __len__(self): return len(self.flat_data)
    def __getitem__(self, idx):
        item = self.flat_data[idx]
        embedding = torch.load(item['embedding_path'])
        question = item['question']; answer = item['answer']
        return embedding, question, answer