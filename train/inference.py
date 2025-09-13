import os
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from train.trainer import VLM  # We still need the VLM class definition for the tokenizer

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

class VLMInference:
    def __init__(self, config_path='config.yml', checkpoint_dir='vlm_checkpoints'):
        print(f"{Colors.HEADER}--- Initializing VLM for Inference ---{Colors.ENDC}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        if torch.backends.mps.is_available(): self.device = "mps"
        elif torch.cuda.is_available(): self.device = "cuda"
        else: self.device = "cpu"
        print(f"Using device: {self.device}")

        self._load_model_and_adapters(checkpoint_dir)
        self._load_tokenizer()
        
        # --- THE FIX IS HERE ---
        # Set both model components to evaluation mode individually.
        self.llm.eval()
        self.projector.eval()
        
        print(f"{Colors.GREEN}--- Initialization Complete ---{Colors.ENDC}")

    def _load_model_and_adapters(self, checkpoint_dir):
        base_llm = AutoModelForCausalLM.from_pretrained(
            self.config['model']['llm_name'],
            dtype=torch.bfloat16
        )
        
        adapter_path = os.path.join(checkpoint_dir, 'best_llm_adapters')
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"LLM adapters not found at: {adapter_path}")
        
        self.llm = PeftModel.from_pretrained(base_llm, adapter_path)
        print("Base LLM and trained adapters loaded successfully.")
        
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.config['model']['vision_embedding_dim'], self.config['model']['projector_hidden_dim']),
            torch.nn.GELU(),
            torch.nn.Linear(self.config['model']['projector_hidden_dim'], self.config['model']['llm_embedding_dim'])
        ).to(self.device).to(torch.bfloat16)
        
        projector_path = os.path.join(checkpoint_dir, 'best_projector.pth')
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector checkpoint not found at: {projector_path}")
        
        self.projector.load_state_dict(torch.load(projector_path, map_location=self.device))
        print("Projector weights loaded successfully.")

        self.llm.to(self.device)

    def _load_tokenizer(self):
        self.tokenizer = VLM.get_tokenizer(self.config['model']['llm_name'])

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
        else: # Fallback if "Answer: " is not in the output
            answer = full_decoded_text.strip()

        return answer

if __name__ == '__main__':
    # --- CONFIGURATION ---
    CHECKPOINT_DIR = 'checkpoints/vlm_checkpoints'
    TEST_DATA_DIR = 'data/final_split_dataset_versatile/test/embeddings'
    
    # --- SCRIPT EXECUTION ---
    try:
        inference_engine = VLMInference(checkpoint_dir=CHECKPOINT_DIR)
    
        # Find a sample from the test set to ask a question about
        sample_embedding_name = '0ac90ac0-37cbb7cc.pt' # You can change this to any file in your test set
        sample_embedding_path = os.path.join(TEST_DATA_DIR, sample_embedding_name)
        
        questions = [
            "Is there a car in the image?",
            "What is the color of the traffic light?",
            "How is the weather in this scene?",
            # "What are the objects in the image?"
        ]
        
        print(f"\n{Colors.BLUE}--- Asking questions about image: {sample_embedding_name} ---{Colors.ENDC}")
        for q in questions:
            answer = inference_engine.answer_question(sample_embedding_path, q)
            print(f"\nQ: {q}")
            print(f"A: {Colors.GREEN}{answer}{Colors.ENDC}")

    except FileNotFoundError as e:
        print(f"\n{Colors.RED}FATAL ERROR: A required file or directory was not found.{Colors.ENDC}")
        print(f"{Colors.YELLOW}{e}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please ensure your checkpoint and test data directories are correct.{Colors.ENDC}")