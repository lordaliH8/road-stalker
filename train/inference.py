import os
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import our custom classes from the trainer module
from train.trainer import VLM, QuantumProjector 

# ANSI color codes
class Colors:
    GREEN = '\033[92m'; BLUE = '\033[94m'; HEADER = '\033[95m'; ENDC = '\033[0m'; YELLOW = '\033[93m'; RED = '\033[91m'

class VLMInference:
    def __init__(self, config_path='config.yml', checkpoint_dir='checkpoints/vlm_checkpoints'):
        print(f"{Colors.HEADER}--- Initializing VLM for Inference ---{Colors.ENDC}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        if torch.backends.mps.is_available(): self.device = "mps"
        elif torch.cuda.is_available(): self.device = "cuda"
        else: self.device = "cpu"
        print(f"Using device: {self.device}")

        # --- CORRECTED INITIALIZATION ORDER ---
        # 1. Load tokenizer first, as we need its size.
        self._load_tokenizer()
        # 2. Then, load the model and adapters, using the tokenizer's info.
        self._load_model_and_adapters(checkpoint_dir)
        
        self.llm.eval()
        self.projector.eval()
        
        print(f"{Colors.GREEN}--- Initialization Complete ---{Colors.ENDC}")

    def _load_tokenizer(self):
        # Use the static helper method to ensure tokenizer is identical to training
        self.tokenizer = VLM.get_tokenizer(self.config['model']['llm_name'])

    def _load_model_and_adapters(self, checkpoint_dir):
        # Step A: Load the BASE language model
        base_llm = AutoModelForCausalLM.from_pretrained(
            self.config['model']['llm_name'],
            dtype=torch.bfloat16
        )
        
        # --- THE DEFINITIVE FIX ---
        # Step B: Resize the token embeddings to match the tokenizer, which now includes '<image>'.
        # This MUST be done before loading the adapters.
        base_llm.resize_token_embeddings(len(self.tokenizer))
        print(f"Resized model vocabulary to {len(self.tokenizer)} to accommodate special tokens.")
        
        # Step C: NOW, load the trained LoRA adapters onto the correctly-sized base model
        adapter_path = os.path.join(checkpoint_dir, 'best_llm_adapters')
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"LLM adapters not found at: {adapter_path}")
        
        self.llm = PeftModel.from_pretrained(base_llm, adapter_path)
        print("Base LLM and trained adapters loaded successfully.")
        
        # Step D: Instantiate and load the projector
        is_qml = self.config.get('qml', {}).get('enabled', False)
        if is_qml:
            print("Instantiating Quantum Projector...")
            self.projector = QuantumProjector(
                input_dim=self.config['model']['vision_embedding_dim'],
                output_dim=self.config['model']['llm_embedding_dim'],
                num_qubits=self.config['qml']['num_qubits']
            )
        else:
            print("Instantiating Classical Projector...")
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(self.config['model']['vision_embedding_dim'], self.config['model']['projector_hidden_dim']),
                torch.nn.GELU(),
                torch.nn.Linear(self.config['model']['projector_hidden_dim'], self.config['model']['llm_embedding_dim'])
            )

        projector_path = os.path.join(checkpoint_dir, 'best_projector.pth')
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector checkpoint not found at: {projector_path}")
        
        self.projector.load_state_dict(torch.load(projector_path, map_location=self.device))
        self.projector.to(self.device).to(torch.bfloat16)
        print("Projector weights loaded successfully.")

        self.llm.to(self.device)

    def answer_question(self, image_embedding_path, question):
        if not os.path.exists(image_embedding_path):
            return f"Error: Embedding file not found at {image_embedding_path}"
            
        embedding = torch.load(image_embedding_path).to(self.device).to(torch.bfloat16).unsqueeze(0)
        prompt = f"<image>\nQuestion: {question}\nAnswer: "
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            projected_embedding = self.projector(embedding)
            prompt_embeddings = self.llm.get_input_embeddings()(tokenized_prompt.input_ids)
            inputs_embeds = torch.cat([projected_embedding.unsqueeze(1), prompt_embeddings[:, 1:, :]], dim=1)
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=50,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )

        full_decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_start_index = full_decoded_text.find("Answer: ")
        if answer_start_index != -1:
            answer = full_decoded_text[answer_start_index + len("Answer: "):].strip()
        else:
            answer = full_decoded_text.strip()

        return answer

if __name__ == '__main__':
    CHECKPOINT_DIR = 'checkpoints/qvlm_checkpoints'
    CONFIG_FILE = 'config.yml'
    TEST_DATA_DIR = 'data/final_split_dataset_versatile/test/embeddings'
    
    try:
        inference_engine = VLMInference(config_path=CONFIG_FILE, checkpoint_dir=CHECKPOINT_DIR)
    
        sample_embedding_name = '000e0252-8523a4a9.pt'
        sample_embedding_path = os.path.join(TEST_DATA_DIR, sample_embedding_name)
        
        questions = [
            "Is there a car in the image?",
            "What is the color of the traffic light?",
            "How is the weather in this scene?",
        ]
        
        print(f"\n{Colors.BLUE}--- Asking questions about image: {sample_embedding_name} ---{Colors.ENDC}")
        for q in questions:
            answer = inference_engine.answer_question(sample_embedding_path, q)
            print(f"\nQ: {q}")
            print(f"A: {Colors.GREEN}{answer}{Colors.ENDC}")

    except FileNotFoundError as e:
        print(f"\n{Colors.RED}FATAL ERROR: A required file or directory was not found.{Colors.ENDC}")
        print(f"{Colors.YELLOW}{e}{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}An unexpected error occurred:{Colors.ENDC}")
        print(f"{Colors.YELLOW}{e}{Colors.ENDC}")