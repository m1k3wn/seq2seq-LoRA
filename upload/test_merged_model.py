from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import Optional, Dict, Any

# Default configurations
DEFAULT_MODEL_NAME = "nidra_v2"
DEFAULT_TEST_INPUT = "Interpret this dream: I had a dream about flying over mountains."

# Optional generation overrides (only used if you want to override generation_config.json)
    # Uncomment and modify any settings you want to override
GENERATION_OVERRIDES: Dict[str, Any] = {
    "temperature": 0.4,
    "max_length": 512,
    "min_length": 150,
    "do_sample": True,
    "num_beams": 8,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 4,
    "length_penalty": 1.2,
    "early_stopping": True
}

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent

def test_model(
    model_name: str = DEFAULT_MODEL_NAME, 
    test_input: str = DEFAULT_TEST_INPUT,
    generation_overrides: Optional[Dict[str, Any]] = None
) -> None:
    """
    Test a merged model with a sample input
    
    Args:
        model_name (str): Name of the merged model
        test_input (str): Input text to test with
        generation_overrides (Dict[str, Any], optional): Override default generation settings
    """
    # Setup paths
    project_root = get_project_root()
    model_path = project_root / "merged_models" / f"{model_name}_merged"
    
    print(f"\nModel Test Configuration:")
    print(f"👻 Loading model from: {model_path}")
    
    # Verify model exists
    if not model_path.exists():
        raise ValueError(f"☠ Model not found at {model_path}. Have you merged it yet?")
    
    # Load model and tokenizer
    print("⏳ Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Prepare generation kwargs
    generation_kwargs = {}
    if generation_overrides:
        print("\n🗂 Using these generation overrides:")
        for key, value in generation_overrides.items():
            print(f"{key}: {value}")
        generation_kwargs.update(generation_overrides)
    
    print("\n🔎 Running inference with test input:")
    print(f"Input: '{test_input}'")
    
    # Tokenize input
    inputs = tokenizer(test_input, return_tensors="pt", truncation=True)
    
    # Generate output using config defaults plus any overrides
    print("\n🔮 Generating interpretation...")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            **generation_kwargs
        )
    
    # Decode and print outputs
    print("\nModel Output(s):")
    print("-" * 80)
    for idx, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        if len(outputs) > 1:
            print(f"\nInterpretation {idx + 1}:")
        print(f"{generated_text}")
        print("-" * 80)
    
    print("\n🎉 Test completed successfully!")

def main():
    try:
        test_model(
            # Use defaults or override as needed:
            model_name=DEFAULT_MODEL_NAME,
            test_input=DEFAULT_TEST_INPUT,
            generation_overrides=GENERATION_OVERRIDES if GENERATION_OVERRIDES else None
        )
    except Exception as e:
        print(f"\n☠ Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()