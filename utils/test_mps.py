import torch

def check_mps_support():
    print(f"\n🧳  PyTorch version: {torch.__version__}")
    print(f"\n💣  MPS available: {torch.backends.mps.is_available()}")
    print(f"\n🧨  MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        print("\n🔥 MPS is available - let's go!")
        device = torch.device("mps")
    else:
        print("\n🙈  MPS not available:")
        if not torch.backends.mps.is_built():
            print("🙊  PyTorch was not built with MPS support")
            print("\n🐵  Try reinstalling PyTorch with: pip install --upgrade torch")
        
        print("\n🐌  Falling back to CPU...")
        device = torch.device("cpu")
    
    return device

if __name__ == "__main__":
    device = check_mps_support()