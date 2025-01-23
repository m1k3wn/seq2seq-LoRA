from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import sys
from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory from upload/merge_base_model.py"""
    # Go up one level from the upload directory to reach project root
    return Path(__file__).resolve().parent.parent

def setup_paths(fine_tuned_model_name: str) -> tuple[Path, Path]:
    """
    Configure paths for model merging
    
    Args:
        fine_tuned_model_name (str): Name of the fine-tuned model
        
    Returns:
        tuple[Path, Path]: Tuple containing (lora_model_path, merged_model_path)
    """
  # Get project root and add to Python path
    project_root = get_project_root()
    sys.path.append(str(project_root))
    
    # Configure paths relative to project root
    lora_model_path = project_root / "outputs" / "models" / fine_tuned_model_name
    merged_model_path = project_root / "merged_models" / f"{fine_tuned_model_name}_merged"
    
    # Create the directory for the merged model if it doesn't exist
    merged_model_path.mkdir(parents=True, exist_ok=False)
    
    return lora_model_path, merged_model_path

def merge_model(
    base_model_name: str,
    lora_model_path: Path,
    merged_model_path: Path
) -> None:
    """
    Merge LoRA weights into base model and save the result
    """
    print(f"🤖 Loading base model: {base_model_name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    
    print(f"👾 Loading LoRA from: {lora_model_path}")
    peft_config = PeftConfig.from_pretrained(lora_model_path)
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    print("🖇 Merging models...")
    merged_model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {merged_model_path}")
    merged_model.save_pretrained(merged_model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(merged_model_path)
    print("🌟 Tokenizer saved successfully")

def main():
    # Configuration
    base_model_name = 'google/flan-t5-base'
    #  Assign model name 
    # * must match fine-tuned model folder and will become output folder name *
    fine_tuned_model_name = "nidra_v1"
    
    # Setup paths
    lora_model_path, merged_model_path = setup_paths(fine_tuned_model_name)
    
    # Print paths for verification
    print("\n🗄 Path Configuration:")
    print(f"Project root: {get_project_root()}")
    print(f"LoRA model path: {lora_model_path}")
    print(f"Merged model will be saved to: {merged_model_path}\n")
    
    # Perform merge
    merge_model(base_model_name, lora_model_path, merged_model_path)
    print("🎉 Model merging completed successfully!")

if __name__ == "__main__":
    main()