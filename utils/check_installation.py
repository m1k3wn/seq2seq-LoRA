# Create a file called check_installation.py with this content:
import torch
import transformers
import accelerate
import numpy as np
import pandas as pd
import wandb
import sklearn

def check_environment():
    print("=== Environment Check ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Numpy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"\nM1 Acceleration:")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS backend: {torch.backends.mps.is_built()}")

if __name__ == "__main__":
    check_environment()