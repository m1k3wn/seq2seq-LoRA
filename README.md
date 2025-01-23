### Summary 

A model-agnostic trianing pipeline optimised for fine-tuning smaller Seq2Seq models on desktop hardware with a small dataset. 

### Hardware
Defaults to using MPS if available, falls back to CPU if not. Can be reconfigured in config.py. Verify with /utils/test_mps.py.   
Currently tested on apple M1 silicone / 24 core GPU / 32gb RAM.

## Project Guide

### Setup:

1. Create a new Python virtual environment:   
    python3 -m venv ml_project_env # Creates a new venv in current local directory with chosen name: ml_project_env (you'll see a new dir with this name appear in root dir)

2. Activate local enrionment:  
    source ml_project_env/bin/activate  # Activates new venv (ml_project_env): Should see (venv) appear in your CLI prompt. 

3. Install all requirements:  
    pip install -r requirements.txt

4. Configure VScode to current virtual environment:   
    Cmd+Shift+P (Mac) # Bring up command palette  
    Type "Python: Select Interpreter"   
    Choose ml_project_env environment  
    (will likely be an older version than you have available globally - many ML packages don't support newest versions yet)

5. Run /utils/environment_check.py to ensure all libraries and packages are correctly installed and versions match
6. Run /utils/test_mps.py to see if MPS is available, or fall back to CPU (boo, slow)

### Convert dataset:
1. Add dataset to /data in JSON format with input/output pairs
2. In config.py change lines 21-22 to match your input-output key names
3. Update input_prompt_template for your task
4. Check converted dataset by running /utils/inspect_dataset.py to see decoded examples, select either Assign training or validation dataset on lines 54-55
5. If needed, you can also run /utils/inspect_tensors.py to inspect any potential errors encoding tensors

### Set configurations
All project configs are located in /src/configs.py; see below for detailed explanations of each. 


### Inference testing: 
Best way is with /utils/inference_test.py : Assign model name at top and input some test inputs, assign various configs and compare results. 
Can also interact with a single model input-by-input by using /utils/cli_inference_test.py

### Folder structure
```plaintext
FINE_TUNING_SEQ2SEQ         # Root
├── data                    # Initial training data (JSON)
│   └── dreams_data.json    
├── merged_models           # Output for merged models
│   ├── nidra_v1_merged          # New models auto-saved with specified name + "_merged"
│   └── nidra_v2_merged    
├── ml_project_env          # Python virtual environment dir
├── outputs                 # Stores fine-tuned models and PyTorch tensor datasets
│   └── models                  # New fine-tuned models saved here with timestamped name  
│   │    ├── model_<timestamp> 
│   │    └── model_<timestamp>
│   ├── train_dataset.pt        # Output dir for PyTorch tensor conversions of training and validation data  
│   └── val_dataset.pt
├── src                         # Main training scripts       
│   ├── ___init__.py                # Marks as Python package - allows exports
│   ├── config.py                   # All key project configurations 
│   ├── data_preprocessing.py       # All data processing logic 
│   ├── prepare_data.py
│   └── train.py
├── test_results
├── upload
│   ├── merge_base_model.py
│   ├── save_spiece_file.py
│   ├── test_merged_model.py
│   ├── test_uploaded_model.py
│   └── upload_to_huggingface.py
├── utils                       # Environment check, MPS check, dataset inspection, inference test
├── wandb                       # Weights and Balances - advanced training diagnostics/reporting if using
├── .env
├── README.md
└── requirements.txt
```

## Working with new datasets
1. In config.py updated the 'source_input_field' and 'source_output_field' to match key names in your training data.
2. Update input_prompt_template for task.

## config.py
All project configurations centralised here. Uses typehints for datatype clarity. 
Configuration Guide balanced for T5 Fine-tuning on m1 apple silicone:

### Directory Structure
base_dir: The root directory of your project  
data_dir: Where your raw data is stored  
output_dir: Where processed files and results are saved   
models_dir: Where trained models are saved   

### Data Configuration
These settings control how your training data is structured:

data_path: Location of your input JSON file containing dream and interpretation pairs  
train_dataset_path: Where the processed training dataset will be saved  
val_dataset_path: Where the processed validation dataset will be saved  
source_input_field: Name of the field in your JSON containing the input   
source_target_field: Name of the field containing the target output  
input_prompt_template: Template that prepends instructions to each input  

### Base Model Settings
Core settings for the model:

model_name: Which pre-trained model to use (currently google/flan-t5-base)  
max_input_length: Maximum length of input text  
max_target_length: Maximum length of output text  

### Training Configuration
Settings that control how the model learns:

train_ratio: What portion of data to use for training (0.9 = 90% training, 10% validation)  
batch_size: How many examples to process at once (2 - kept small for memory constraints)  
gradient_accumulation_steps: Helps simulate larger batch size with limited memory  
num_epochs: How many times to go through the entire dataset  
learning_rate: How big of learning steps to take (between 2e4 and 5e-4 is good for T5 base)  
weight_decay: Prevents overfitting. but too high model will underfit (0.015-0.02 is sweet spot for T5)  
warmup_steps: Gradually increases learning rate to prevent unstable start   
lr_scheduler_type: How learning rate changes over time (uses cosine schedule)  
max_grad_norm: Prevents extreme parameter updates   

### Evaluation and Logging
Settings for tracking progress:

evaluation_strategy: When to evaluate ("steps" means every n steps)  
eval_steps: How often to evaluate (ie every 50 steps)  
save_strategy: When to save checkpoints ("steps" means every n steps)  
save_steps: How often to save (ie every 50 steps)  
logging_steps: How often to log metrics (ie every 25 steps)  
save_total_limit: Maximum number of checkpoints to keep (ie 3)  
early_stopping_patience: Stop if no improvement for n evaluations (ie 3)  
eval_batch_size: Batch size during evaluation (ie 2)  
max_eval_samples: Maximum examples to use in evaluation (ie 150)  
eval_accumulation_steps: Similar to training accumulation but for eval (ie 4)  
report_to: Where to send training logs (tensorboard)  
metric_for_best_model: Which metric determines best model (eval_loss)  
load_best_model_at_end: Whether to restore best checkpoint at end (true)  

### LoRA Settings
Parameters for Low-Rank Adaptation (efficient fine-tuning):

lora_r: Rank of LoRA matrices; determines how many parameters from base model are adjusted  
lora_alpha: Scaling factor- controls how strong LoRA modifications are.  
(rank:alpha ratio wants to be kept between 1:2 to 1:4)  
lora_dropout: Helps prevent overfitting   
lora_target_modules: Which parts of model to modify (using attention components; q,k,v,o)  
-  Q: Query - How model formulates its questions about the inputs
- K: Key - How model organises information for lookup (like key words basically)
- V: Value - How model stores/retrieves actual content 
- O: Output - How model formulates final attention output
- LoRA can additionally target many more attention layers, the above seemed to work well for fine-tuning on small datasets. 

### Hardware Settings (M1 Mac Specific)
Optimizations for Apple Silicon:

fp16: disabled 16-bit floating point precision   
bf16: Alternative 16-bit format enabled  
no_cuda: Disable CUDA GPU support   
use_mps_device: Enable Apple Metal GPU support  

### Generation Settings
Controls how the model generates interpretations:

gen_max_length: Maximum length of generated text (512 tokens)  
gen_min_length: Minimum length of generated text (256 tokens)  
gen_num_beams: Number of beams for beam search (8 beams)  
gen_temperature: Randomness in generation (0.4 - lower is more focused  )
gen_do_sample: Whether to use random sampling (true)  
gen_top_p: Cumulative probability cutoff for sampling (0.95)  
gen_use_cache: Cache computations for efficiency (true)  
gen_length_penalty: Encourages longer/shorter outputs (1.2 - slight preference for longer)  
gen_repetition_penalty: Discourages word repetition (3.0)  
gen_no_repeat_ngram_size: Size of phrases not to repeat (4 tokens)  
<br>

## data_preprocessing.py
Main encapsulated data-handling class managing: load, verify, shuffle, split, create, convert and save tensor datasets.  
 Retrieves JSON training data input/output names and initialises class instance with correct tokenizer and pad_token/eos_token using  model_name and source_input_field / source_output_field from config.py

## prepare_data.py
Script to prepare and save data: loads, splits, creates, validates and saves dataset into tensor format. 

## train.py
Main training function. 

## /utils
Contains various utility scripts for project pipeline: 

- environment_check : Test environment setup and library/package version match  
- test_mps: Tests if MPS is available and defaults to torch using CPU if not   
- inspect_dataset: converts .pt tensor dataset back to tokens and text for inspection  
- inspect_tensors: converts .pt tensor data back to preview of tensors for additional inspection  
- uk_spelling: Can be used to replace any US spelling with UK as needed  
- inference_test: Main script to interact with and inference test a fine-tuned models using multiple params/inputs  
- cli_inference_test: Simplified way to interact with a model via CLI using 1 generation_configs set. 

## / upload
Scripts to merge fine-tuned model with base model, test merged model, generation spiece.model files if needed, upload a merged model to huggingface and test an uploaded model.
### merge_base_model
Script to merge base model with your fine-tuned model configs. Adjust lines 61 + 63. 
### test_merged_model
To test your merged model before pushing to huggingface. Adjust name and test input as needed. Uncomment generation_configs if needed to experiment with different parameters. 

### upload_to_huggingface
Update lines 63-66 of 'main' function accordginly. When running script you'll be prompted to enter a valid HF access token in CLI. 



