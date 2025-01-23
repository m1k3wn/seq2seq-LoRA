import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

# Imports
import sys
import torch
from typing import Dict, Any
import numpy as np
from datetime import datetime

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    default_data_collator,
    # GenerationConfig,
)
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path
import torch.serialization
import warnings

# Memory monitoring debug 
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9} GB")

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_preprocessing import DreamAnalysisDataset
from src.config import TrainingConfig

torch.serialization.add_safe_globals([T5TokenizerFast, DreamAnalysisDataset])

# Safe training evaluation phase (using forward pass and memory clears)
class SafeEvaluator(TrainerCallback):
    def __init__(self, trainer, tokenizer, max_eval_samples=None):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.max_eval_samples = max_eval_samples
        
    def print_memory_usage(self, stage):
        print(f"\nMemory usage at {stage}:")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU Cache: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        if hasattr(torch.mps, 'current_allocated_memory'):
            print(f"MPS Memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
        
    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            
    def evaluate_with_safety(self, ignore_keys=None, **kwargs):
        # Suppress ugly warning message 
        warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True") # Running eval with torch.no_grad(). gradient checkpointing still enabled for memory efficiency
        try:
            print("\n=== 🔮 Starting Safe Evaluation 🔮 ===")
            self.print_memory_usage("start")
            
            # Get evaluation dataset
            eval_dataset = self.trainer.eval_dataset
            if self.max_eval_samples:
                eval_dataset = eval_dataset[:self.max_eval_samples]
                print(f"\nUsing {self.max_eval_samples} samples for evaluation")
            
            self.clear_memory()
            self.print_memory_usage("after clear")
            
            eval_dataloader = self.trainer.get_eval_dataloader(eval_dataset)
            device = self.trainer.model.device
            
            print("\n⏱ Running evaluation loop...")
            total_loss = 0
            num_batches = 0
            
            for step, batch in enumerate(eval_dataloader):
                # For GPU debugging 
                # if step % 10 == 0:
                    # print(f"Processing batch {step}")
                    # self.print_memory_usage(f"batch {step}")
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Use model's forward pass instead of generate
                with torch.no_grad():
                    outputs = self.trainer.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    num_batches += 1
                
                self.clear_memory()
            
            # Calculate average loss
            eval_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            metrics = {
                "eval_loss": eval_loss,
                "eval_samples": len(eval_dataset),
            }
            
            print("\n=== Evaluation Complete ===")
            print("Metrics:", metrics)
            self.print_memory_usage("end")
            
            return metrics
            
        except Exception as e:
            print(f"\n!!! Evaluation failed: {str(e)}")
            self.print_memory_usage("error")
            raise e  # Re-raise to stop training if evaluation fails

# Provides detailed training progress
class EnhancedProgressCallback(TrainerCallback):
    """Enhanced callback for detailed training progress"""
    def __init__(self):
        self.best_loss = float('inf')
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n{'='*50}")
        print(f"Starting epoch {state.epoch + 1}/{args.num_train_epochs}")
        print(f"{'='*50}")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            if hasattr(state, 'loss'):
                print(f"Step {state.global_step}: loss = {state.loss:.4f}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            current_loss = metrics.get("eval_loss", float('inf'))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                print(f"\n🌟 New best validation loss: {current_loss:.4f}")
            
            print(f"\nValidation metrics:")
            for key, value in metrics.items():
                print(f"- {key}: {value:.4f}")

# Enhanced model checkpointing and save logs
class CustomSaveCallback(TrainerCallback):
    """Callback for enhanced checkpointing information"""
    def on_save(self, args, state, control, **kwargs):
        # Safely retrieve loss from log_history
        loss_history = getattr(state, "log_history", None)
        if loss_history and isinstance(loss_history, list) and len(loss_history) > 0:
            last_loss_entry = loss_history[-1]
            loss = last_loss_entry.get("loss", None)
            if loss is not None:
                print(f"\n⏱ Last logged loss: {loss:.4f}")
            else:
                print("Loss is not available in the last log entry.")
        else:
            print("Loss history is unavailable.")
        return control

# Main compute - Processes model predictions and labels / converts tensorts to NumPy arrays
def compute_metrics(eval_pred, tokenizer) -> Dict[str, float]:
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    # Debug: Inspect raw predictions and labels
    print(f"Predictions type: {type(predictions)}, shape: {np.shape(predictions) if isinstance(predictions, np.ndarray) else 'N/A'}")
    print(f"Labels type: {type(labels)}, shape: {np.shape(labels) if isinstance(labels, np.ndarray) else 'N/A'}")

    # Converts tensors to standardised numpy arrays. Prevents unecessary memory retention
    def to_numpy(tensor_or_array):
        if torch.is_tensor(tensor_or_array):
            return tensor_or_array.detach().cpu().numpy()
        elif isinstance(tensor_or_array, np.ndarray):
            return tensor_or_array
        elif isinstance(tensor_or_array, tuple):
            return to_numpy(tensor_or_array[0])
        else:
            raise TypeError(f"☠ Unsupported type for conversion: {type(tensor_or_array)}")

    predictions = to_numpy(predictions)
    labels = to_numpy(labels)

    if len(predictions.shape) == 3 and predictions.shape[-1] == tokenizer.vocab_size:
        predictions = np.argmax(predictions, axis=-1)
        print(f"🧮 Predictions converted to token IDs: shape {predictions.shape}")

    def remove_padding(arr, pad_token_id):
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        elif torch.is_tensor(arr):
            arr = arr.cpu().numpy().tolist()

        result = []
        for seq in arr:
            if isinstance(seq, list):
                cleaned_seq = [int(token) for token in seq if isinstance(token, (int, float)) and token != pad_token_id]
                result.append(cleaned_seq)
            else:
                raise ValueError(f"\n☠ Unexpected sequence type: {type(seq)} - {seq}")
        return result

    predictions_no_pad = remove_padding(predictions, tokenizer.pad_token_id)
    labels_no_pad = remove_padding(labels, tokenizer.pad_token_id)
    
    # Assign decoded predictions and labels before referencing them
    decoded_preds = tokenizer.batch_decode(predictions_no_pad, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_no_pad, skip_special_tokens=True)

    # Debugging: Print decoded predictions and labels
    # print(f"Decoded Predictions: {decoded_preds[:2]}")
    # print(f"Decoded Labels: {decoded_labels[:2]}")

    avg_pred_length = sum(len(pred.split()) for pred in decoded_preds) / len(decoded_preds)
    avg_label_length = sum(len(label.split()) for label in decoded_labels) / len(decoded_labels)
    length_ratio = avg_pred_length / avg_label_length if avg_label_length > 0 else 0

    return {
        "avg_pred_length": avg_pred_length,
        "avg_label_length": avg_label_length,
        "length_ratio": length_ratio,
    }

# Dataset validation
def validate_datasets(train_dataset, val_dataset, tokenizer):
    """Perform quality checks on datasets before training"""
    print("\n🔬 Validating datasets...")
    
    # Check for empty sequences
    empty_train = sum(1 for item in train_dataset if item['labels'].sum() == 0)
    empty_val = sum(1 for item in val_dataset if item['labels'].sum() == 0)
    
    if empty_train > 0 or empty_val > 0:
        print(f"⚠️  Warning: Found {empty_train} empty training sequences and {empty_val} empty validation sequences")
    
    # Check sequence lengths
    train_lengths = [item['input_ids'].ne(tokenizer.pad_token_id).sum().item() for item in train_dataset]
    val_lengths = [item['input_ids'].ne(tokenizer.pad_token_id).sum().item() for item in val_dataset]
    
    print("\n🔎 Sequence length statistics:")
    print(f"Training:")
    print(f"- Average length: {sum(train_lengths)/len(train_lengths):.1f} tokens")
    print(f"- Max length: {max(train_lengths)} tokens")
    print(f"- Min length: {min(train_lengths)} tokens")
    print(f"\nValidation:")
    print(f"- Average length: {sum(val_lengths)/len(val_lengths):.1f} tokens")
    print(f"- Max length: {max(val_lengths)} tokens")
    print(f"- Min length: {min(val_lengths)} tokens")

# Loads pre-trained base model, confgiures device, enabled gradient checkpoints, applies LoRA
def setup_model_and_tokenizer(config: TrainingConfig):
    print("\n🗄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Debugging to Ensure pad token is set / set to eos token
    # if tokenizer.pad_token is None:
    #     print("Setting pad token...")
    #     tokenizer.pad_token = tokenizer.eos_token
    # print(f"Pad token: {tokenizer.pad_token}")
    # print(f"Pad token ID: {tokenizer.pad_token_id}")
    
    print("\n🤖 Loading base model...")
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n📺 Using device: {device}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
    )
    model = model.to(device)
    
    print("\n🧮 Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
 
    print("\n🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules
    )
    
    print("\n🔩 Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

torch.serialization.add_safe_globals([DreamAnalysisDataset])

# Loads datasets
def load_datasets(config: TrainingConfig):
    """Load and validate the preprocessed datasets"""
    print("\n📚 Loading datasets...")
    try:
        train_dataset = torch.load(config.train_dataset_path)
        val_dataset = torch.load(config.val_dataset_path)
        
        print(f"✓ Loaded {len(train_dataset)} training examples")
        print(f"✓ Loaded {len(val_dataset)} validation examples")
        return train_dataset, val_dataset
    except Exception as e:
        raise RuntimeError(f"☠ Error loading datasets: {e}")


# Main training function
def train(config: TrainingConfig, model, tokenizer):
    """Main training function with enhanced monitoring and validation"""
    print("\n🚀 Initializing training...")
    
    # Assign default generation configs
    model.generation_config = config.get_generation_config(tokenizer)
    
    # Create timestamped directory for each new model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = config.models_dir / f"model_{timestamp}"
    model_save_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Model save path: {model_save_path}")
    
    # Load and validate datasets
    train_dataset, val_dataset = load_datasets(config)
    validate_datasets(train_dataset, val_dataset, tokenizer)
    
    # # Initialize trainer with updated arguments
    training_args = TrainingArguments(
        output_dir=str(model_save_path),
        **config.get_training_arguments()
    )

    # Assign all configs to trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        callbacks=[
            EnhancedProgressCallback(),
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
            CustomSaveCallback()
        ]
    )
    
    # Add safe evaluator
    safe_evaluator = SafeEvaluator(trainer, tokenizer, max_eval_samples=config.max_eval_samples)
    trainer.evaluate = safe_evaluator.evaluate_with_safety
    
    print("\n🎛 Starting training...")
    try:
        trainer.train()
        
        print("\n💾 Saving final model...")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        print(f"\n✅ Model and adapter saved to {model_save_path}")
        
        print("\n📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        print("\n⏱ Final evaluation metrics:")
        for key, value in eval_results.items():
            print(f"- {key}: {value:.4f}")
            
        return trainer
        
    except Exception as e:
        print(f"\n☠ Error during training: {e}")
        raise

# Fifth: Main Execution
def main():
    """Main execution function"""
    try:
        print("\n🧠 Initializing model training...")
        config = TrainingConfig()

        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)

        trainer = train(config, model, tokenizer)
        print("\n✨ Training completed successfully!")
        return trainer
    except Exception as e:
        print(f"\n☠ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()