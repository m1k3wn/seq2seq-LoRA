import torch
from pathlib import Path

# Dynamically set paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
outputs_dir = parent_dir / "outputs" 

# Define dataset paths
train_dataset_path = outputs_dir / "train_dataset.pt"
val_dataset_path = outputs_dir / "val_dataset.pt"

# Load datasets
try:
    # Load training and validation datasets
    train_data = torch.load(train_dataset_path)
    val_data = torch.load(val_dataset_path)
    
    # Print dataset sizes
    print(f"\n📚 Training dataset size: {len(train_data)}")
    print(f"\n 📒Validation dataset size: {len(val_data)}")
    
    # Print an example item from training data
    print("\n🔎 Example training data item:")
    print(train_data[0])
except FileNotFoundError as e:
    print(f"\n☠ Dataset file not found: {e}")
except Exception as e:
    print(f"\n☠ Error loading datasets: {e}")
