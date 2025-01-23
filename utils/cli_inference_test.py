from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Test a fine-tuned model in CLI before merging with base model 
class DreamInterpreter:
    def __init__(self):
        # Set up paths
        self.base_dir = Path(__file__).parent.parent
        self.model_dir = self.base_dir / "outputs" / "models" / "nidra_v1" # Replace with current model name
        
        # Set device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print("Loading model and tokenizer...")
        self.base_model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name)
        
        # Load adapter
        print("Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(self.base_model, self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def interpret_dream(self, dream_text: str) -> str:
      
        input_text = ("{dream_text}"
        )

        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Add naughty-list words as needed
        bad_phrases = [
            "Please share", "comments are confidential", 
            "comment section", "comments",
            "Please feel free", "Thank you for taking the time", 
            "I'd love to hear your thoughts",
            "contact me", "Editor-in-Chief", "DMCA",
            "dream interpreter", "dream interpretation","dream interpreters",
            "Thank you for your time", "Best wishes", 
            "Are you kidding me", "Goodbye", "dream writer",
            "my wife", "another dreamer", "dreamers response",
            "this world", ".com", "Please give me a call", "I", "my"
        ]
        bad_words_ids = [self.tokenizer.encode(phrase, add_special_tokens=False) for phrase in bad_phrases]
    
        with torch.no_grad():
          outputs = self.model.generate(
            # Update generation configs as needed
            **inputs,
            max_length=300,
            min_length=150,        
            num_beams=6,            
            temperature=0.5,     
            top_p=0.85,  
            do_sample=True,        
            repetition_penalty=3.0,  
            no_repeat_ngram_size=4,  
            early_stopping=True,
            length_penalty=1.2,     
            bad_words_ids=bad_words_ids
        )
        
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    interpreter = DreamInterpreter()
    
    print("\nEnter your dream (or 'quit' to exit):")
    while True:
        dream = input("\nDream: ")
        if dream.lower() == 'quit':
            break
        
        print("\nGenerating interpretation...\n")
        interpretation = interpreter.interpret_dream(dream)
        print(f"Interpretation:\n{interpretation}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()