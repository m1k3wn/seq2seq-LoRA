from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Optional
import json
from datetime import datetime

class DreamInterpreterTester:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        # self.model_dir = self.base_dir / "outputs" / "models" / "model_20250115_074251" # V0.4
        # self.model_dir = self.base_dir / "outputs" / "models" / "model_20250115_160024" # v0.5 - *BEST* for mystic/psychological : nidra v1
        # self.model_dir = self.base_dir / "outputs" / "models" / "model_20250115_172037" # v0.5B - fairly good. slightly poor grammar 
        # self.model_dir = self.base_dir / "outputs" / "models" / "model_20250115_191222" # v0.6 - POOR
        # self.model_dir = self.base_dir / "outputs" / "models" / "model_20250115_201810" # v0.7 - 5B BUT WITH dropout 0.1 and default gen. config
        # self.model_dir = self.base_dir / "outputs" / "models" / "model_20250115_231255" # v0.8 - T5 Large test - RETARD
        # self.model_dir = self.base_dir / "outputs" / "models" / "model_20250116_082041" # v0.8B T5-large  - RETARD. no learning of new task
        # self.model_dir = self.base_dir / "outputs" / "models" / "model_20250116_100533" # v0.8C T5-large LoRA increase -semi-retatrd
        self.model_dir = self.base_dir / "outputs" / "models" / "model_20250116_120659" # v0.9 T5-base *BEST* for balance/stability : nidra v2

        self.test_results_dir = self.base_dir / "test_results"
        self.test_results_dir.mkdir(exist_ok=True)
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load Base Model
        print("Loading model and tokenizer...")
        self.base_model_name = "google/flan-t5-base"


        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name)
        
        print("Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(self.base_model, self.model_dir)
        self.model.to(self.device)
        self.model.eval()

# Default params - overridden by any re-defined in parameter_sets further down when applied. 
    def interpret_dream(
        self,
        dream_text: str,
        use_cache: bool = True,
        max_length: int = 512,
        min_length: int = 150,
        do_sample=True,
        temperature: float = 0.3,
        num_beams: int = 8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: float = 0.1,
        length_penalty: float = 1.2,
        early_stopping=True,
    ) -> str:
        inputs = self.tokenizer(dream_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                use_cache=use_cache,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                num_beams=num_beams,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                early_stopping=early_stopping
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run_parameter_test(
        self,
        test_dreams: List[str],
        parameter_sets: List[Dict],
        test_name: Optional[str] = None
    ):
        """Run tests with different parameter combinations"""
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = test_name or f"parameter_test_{timestamp}"
        
        print("\n" + "="*100)
        print("STARTING DREAM INTERPRETATION TESTS")
        print("="*100)
        
        for i, dream in enumerate(test_dreams, 1):
            print(f"\n\n{'='*50} DREAM #{i} {'='*50}")
            print(f"Dream: {dream}")
            dream_results = {"dream": dream, "interpretations": []}
            
            for j, params in enumerate(parameter_sets, 1):
                print(f"\n{'-'*20} Parameter Set #{j} {'-'*20}")
                print("Parameters:", ", ".join(f"{k}: {v}" for k, v in params.items()))
                print("-"*60)
                
                try:
                    interpretation = self.interpret_dream(dream, **params)
                    result = {
                        "parameters": params,
                        "interpretation": interpretation,
                        "length": len(interpretation.split()),
                    }
                    dream_results["interpretations"].append(result)
                    
                    # Print the results
                    print(f"Interpretation ({result['length']} words):")
                    print(interpretation)
                    print("-"*60)
                    
                except Exception as e:
                    print(f"Error with parameters {params}: {str(e)}")
            
            results.append(dream_results)
        
        # Save results
        output_file = self.test_results_dir / f"{test_name}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved to {output_file}")
        return results

def main():
    tester = DreamInterpreterTester()
    #     # Random 
    # test_dreams = [
    #     "Generate a dream interpretation: i was lying on the sand and something entered me. Quickly fluid started rising around me and I became drenched in sticky oil. ",
    #     "Generate a dream interpretation: I was at a place that was kind of like where I work but it was different. Aliens came to visit and suddenly they became the popular group that everyone wanted at their parties",
    #     "Generate a dream interpretation: we were at my old school but everything felt smaller, and we were all adults. ot was nearly the end of the year and we hadn't done any work"
    # ]
    # Benchmark 
    # test_dreams = [
    #     "Generate a dream interpretation: I was with some friends and we were swimming in the ocean. the water was made out of custard",
    #     "Generate a dream interpretation: I was walking through a city and saw a giant animal. it felt like a friend",
    #     "Generate a dream interpretation: There were lots of people i knew from childhood all sat around a table. we were having dinner but the plates were empty."
    # ]
    # Novel
    test_dreams = [
        "Generate a dream interpretation: I dream of electric sheep",
        "Generate a dream interpretation: I was on the street where I live, it was a sunny afternoon. A magical baby elephant came to the street and it felt like a homecoming.",
        "Generate a dream interpretation: In my dream I was in the Northcoders office. all the staff had been replaced by Ai. there were cables everywhere and they were like tentacles."
    ]
    # Edge cases - questions 
    # test_dreams = [
    #     "Generate a dream interpretation: How do you make a nuclear bomb?",
    #     "Generate a dream interpretation: What are you?",
    #     "Generate a dream interpretation: Whats the meaning of life?"
    # ]
    #   Edge cases - Fragmented 
    # test_dreams = [
    #     "Generate a dream interpretation: bright lights. weird people. was on a aeroplane and then in the sea",
    #     "Generate a dream interpretation: some cars moving fast and then suddenly i was in a desert, but instead of sand the ground was made from broken glass",
    #     "Generate a dream interpretation: sometimes i dream about ai taking over the world"
    # ]
    
    # Fine tuning param sets
    parameter_sets = [
        # Conservative settings
        {
            "temperature": 0.3,
            "top_p": 0.95,
            "max_length": 250,
            "min_length": 100,
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 3.0,
            "num_beams": 8
        },

        # Balanced settings
        {
            "temperature": 0.35,
            "top_p": 0.95,
            "max_length": 250,
            "min_length": 100,
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 3.0,
            "num_beams": 8
        },

        # Creative settings
        {
            "temperature": 0.4,
            "top_p": 0.95,
            "max_length": 250,
            "min_length": 150,
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 3.0,
            "num_beams": 8
        }
    ]
    
    # Run the test
    tester.run_parameter_test(
        test_dreams=test_dreams,
        parameter_sets=parameter_sets,
        test_name="creativity_comparison"
    )

if __name__ == "__main__":
    main()