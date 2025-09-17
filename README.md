# Visual Language Model (VLM) Framework for Simple VQA on City Scenery Images
This repository provides a Visual Language Model (VLM) framework for training a mini VLM on the BDD100k dataset for simple Visual Question Answering (VQA) tasks on city scenery images. The framework supports both classical MLP and quantum neural network (QNN) based projectors for translating image embeddings to the language model.
## Prerequisites
### Download the Dataset
1.  Download the BDD100k dataset.
2.  Place the dataset under the `/data` directory in the root of the project.
### Set Up Python Environment
1.  Create a Python virtual environment.
2.  Install dependencies by running:
    ```sh
    pip install -r requirements.txt
    ```
## Data Preparation
### Merge and Simplify Labels
Run the following command to merge the dataset labels and create a simplified version for VQA tasks:
```sh
python -m data_wizard.prep
```
### Build Dataset
Run the following command to:
*   Generate image embeddings using `OpenAI/CLIP`.
*   Create versatile QA pairs for the dataset.
*   Split the dataset into train, test, and validation sets, along with corresponding `qa.json` files and image embeddings.
```sh
python -m data_wizard.build_dataset
```
### Analyze Dataset
To analyze the refined dataset, run:
```sh
python -m data_wizard.analyze_qa
```
## Training Configuration
Before training, configure the `config.yml` file in the root directory. Below is a sample configuration:
```yaml
# --- Paths and Directories ---
dataset_path: "data/final_split_dataset_versatile"
checkpoint_dir: "checkpoints/qvlm_checkpoints"
# --- Model Architecture ---
model:
  llm_name: "google/gemma-2b"
  vision_embedding_dim: 768
  llm_embedding_dim: 2048
  projector_hidden_dim: 1024
# --- LoRA (PEFT) Configuration ---
lora:
  r: 32 # INCREASED: Give the adapter more capacity to learn.
  lora_alpha: 64 # CONVENTION: Keep alpha at 2x the rank.
  lora_dropout: 0.05
# --- Training ---
training:
  epochs: 5 # INCREASED: Train for more epochs to allow for convergence.
  batch_size: 4
  learning_rate: 2e-5 # START HERE: A safe, small learning rate.
  max_grad_norm: 1.0
  mask_prompt_labels: True
# --- Sampling ---
sampling:
  max_train_samples: 12000 # CRITICAL: Train on the full dataset now.
  max_val_samples: 2000
# --- QML ---
qml:
  enabled: true
  num_qubits: 16
```
You can adjust the training configuration to suit your preferences, and the training module will handle the changes.
## Training
### Start Training
Run the training interface to begin training:
```sh
python -m train.train_interface
```
This will:
*   Start the training process.
*   Create a training plan JSON file for analysis and health checks.
*   Save the latest checkpoint in the `checkpoint_dir` specified in `config.yml`.
### Health Check
To validate the training pipeline, run a single batch and check for meaningful results or errors:
```sh
python -m train.health_check
```
## Evaluation
To evaluate the trained model, run the following command to perform inference on the latest checkpoint and generate a JSON summary of metrics and failed QAs:
```sh
python -m train.eval
```
## Training Approaches
This repository supports two approaches for training the VLM on the BDD100k dataset:
### Classical MLP Projector
Use the `main` branch for training with a classical MLP as the projector between image embeddings and the language model.
### Quantum Neural Network (QNN) Projector
Use the `q_projector_injection` branch to train with a QNN-based projector for the translation between image embeddings and the language model.
The training instructions remain the same for both branches.

