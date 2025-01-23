from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch
from transformers import GenerationConfig, AutoTokenizer

@dataclass
class TrainingConfig:
    # Base paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    output_dir: Path = base_dir / "outputs"
    models_dir: Path = output_dir / "models" 
    
    # Data paths
    data_path: Path = data_dir / "dreams_data.json"
    train_dataset_path: Path = output_dir / "train_dataset.pt"
    val_dataset_path: Path = output_dir / "val_dataset.pt"
    
    # Training data fields
    source_input_field: str = "dream"    # update the JSON input field name
    source_target_field: str = "response" # update the JSON response field name
    # Add system-prompt prepend to training data inputs
    input_prompt_template: str = "Generate a dream interpretation: {}"  # Don't remove {}

    # Model configuration
    model_name: str = "google/flan-t5-base" # Sets base model and tokenizer
    max_input_length: int = 256 
    max_target_length: int = 512
    
    # Training parameters
    train_ratio: float = 0.9 # Ratio of training to evaluation data
    batch_size: int = 2  # Keep low if memory is a constraint
    gradient_accumulation_steps: int = 6  # Compensate for smaller batch but still allow nuance
    num_epochs: int = 6  #  T5 Base 
    learning_rate: float = 5e-4 # T5 Base 
    weight_decay: float = 0.02 # Higher values (like 0.05) = stronger regularization / 0.015-0.02 is sweet spot for T5 base
    warmup_steps = 40  # Gradually increases learning rate from very small to target value / Helps prevent unstable training at the start was 20 for T5 base
    lr_scheduler_type: str = "cosine" 
    max_grad_norm = 0.5   # Stabilise gradients

    # Evaluation and logging configuration
    evaluation_strategy: str = "steps"
    eval_steps: int = 50  # More frequent evaluation
    save_strategy: str = "steps"
    save_steps: int = 50  # Match eval steps
    logging_steps: int = 25  # More frequent logging
    save_total_limit: int = 3
    early_stopping_patience: int = 3
    eval_batch_size: int = 2
    max_eval_samples: int = 150
    eval_accumulation_steps: int = 4
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    metric_for_best_model: str = "eval_loss"
    load_best_model_at_end: bool = True
    
    # LoRA configuration 
    lora_r: int = 8  # Rank: training capacity (higher r=more params moodified)
    lora_alpha: int = 16  # Scaling: controls impact of LoRA mods. maintain r/alpha ratio of 1:2 - 1:4 . Higher alpha = stronger modifications to base model
    lora_dropout: float = 0.15  # Higher dropout can help prevent overfitting, too low and model doesn't learn
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q",
        "k",
        "v",
        "o"
    ]) 
    
    # M1 Mac specific hardware settings
    fp16: bool = False  # Disable mixed precision
    bf16: bool = True  # Toggle bfloat16
    no_cuda: bool = True  # Disable CUDA 
    use_mps_device: bool = True  # Enable Metal Performance Shaders

    # Sets default Generation Configs; applied during inference, via generation_config.json; not baked into model
    gen_max_length: int = 512
    gen_min_length: int = 256
    gen_num_beams: int = 8
    gen_temperature: float = 0.4
    gen_do_sample: bool = True
    gen_top_p: float = 0.95
    gen_use_cache: bool = True
    gen_length_penalty: float = 1.2
    gen_repetition_penalty: float = 3.0
    gen_no_repeat_ngram_size: int = 4

    # Naughty words list
    # Ensures each TrainingConfig instance gets a new copy of the list
    bad_words_ids: List[str] = field(default_factory=lambda: [
        "Please share", "comments are confidential", 
        "comment section", "comments",
        "Please feel free", "Thank you for taking the time", 
        "I'd love to hear your thoughts", "find us at",
        "contact me", "Editor-in-Chief", "DMCA",
        "dream interpreter", "dream interpretation", "dream interpreters",
        "Thank you for your time", "Best wishes", 
        "Are you kidding me", "Goodbye", "dream writer",
        "my wife", "another dreamer", "dreamers response",
        "this world", ".com", "Please give me a call", "I", "my", "I'm",
        "Acupressure", "acoustic"
    ])

    # Creates and returns default generation configs for trained model (can be reassigned during inference)
    def get_generation_config(self, tokenizer: AutoTokenizer) -> GenerationConfig:

        bad_words_ids: List[List[int]] = [
            tokenizer.encode(phrase, add_special_tokens=False) 
            for phrase in self.bad_words_ids
        ]
            
        return GenerationConfig(
            max_length=self.gen_max_length,
            min_length=self.gen_min_length,
            num_beams=self.gen_num_beams,
            temperature=self.gen_temperature,
            do_sample=self.gen_do_sample,
            top_p=self.gen_top_p,
            use_cache=self.gen_use_cache,
            length_penalty=self.gen_length_penalty,
            repetition_penalty=self.gen_repetition_penalty,
            no_repeat_ngram_size=self.gen_no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
        )


    def __post_init__(self):
        """Create necessary directories and set model-specific configurations"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Validate training ratio and batch size
        if self.train_ratio <= 0 or self.train_ratio >= 1:
            raise ValueError("🙈 train_ratio must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("🙊 batch_size must be at least 1")

    def get_training_arguments(self):
        """Return training arguments as a dictionary"""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"📺 Using device: {device}")

        return {
            # "output_dir": str(self.output_dir),
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler_type": self.lr_scheduler_type, 
            "logging_steps": self.logging_steps,
            "evaluation_strategy": "steps",
            "eval_steps": self.eval_steps,
            "eval_accumulation_steps": self.eval_accumulation_steps,
            "per_device_eval_batch_size": self.eval_batch_size,
            "save_strategy": "steps",
            "save_steps": self.save_steps,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            # M1 Mac specific settings
            # "fp16": False,
            "bf16": True, # Faster training and MPS compatible!
            # "no_cuda": True,
            "use_mps_device": True
        }
   