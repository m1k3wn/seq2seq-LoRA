from pathlib import Path
from huggingface_hub import HfApi, login
import sys



def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent

def upload_to_huggingface(
    model_name: str,
    hf_model_name: str,
    username: str,
    private: bool = False
) -> None:
    """
    Upload a merged model to Hugging Face Hub
    
    Args:
        model_name (str): Local model name (e.g., 'nidra_v1')
        hf_model_name (str): Name for the model on HuggingFace
        username (str): Your Hugging Face username
        private (bool, optional): Whether to make the repo private. Defaults to False.
    """
    # Setup paths
    project_root = get_project_root()
    merged_model_path = project_root / "merged_models" / f"{model_name}_merged"
    
    print(f"\nModel Upload Configuration:")
    print(f"\n🤖 Loading model from: {merged_model_path}")
    
    # Verify model exists
    if not merged_model_path.exists():
        raise ValueError(f"☠ Model not found at {merged_model_path}. Have you merged it yet?")
    
    # Login to Hugging Face
    print("\n🔑 Authenticating with Hugging Face...")
    login()  # This will prompt for your token if not already logged in
    
    # Setup repository
    repo_id = f"{username}/{hf_model_name}"
    print(f"\n🗄 Creating/accessing repository: {repo_id}")
    
    # Use the api instance to create the repo
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    
    # Upload model and tokenizer
    print("\n🚀 Uploading model to Hugging Face (this might take a while)...")
    api.upload_folder(
        folder_path=str(merged_model_path),
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"\n🎉 Upload complete! Your model is now available at: https://huggingface.co/{repo_id}")
    if private:
        print("\n🔒 Note: Your repository is private. You'll need to grant access to others who want to use it.")

def main():
    # Configuration - Edit these values
    MODEL_NAME = "nidra_v2"        # Local model name (from merged_models directory)
    HF_MODEL_NAME = "nidra-v2"     # Name for the model on HuggingFace
    HF_USERNAME = "m1k3wn"         # Your HuggingFace username
    PRIVATE_REPO = False           # Set to True if you want a private repository
    
    try:
        upload_to_huggingface(
            model_name=MODEL_NAME,
            hf_model_name=HF_MODEL_NAME,
            username=HF_USERNAME,
            private=PRIVATE_REPO
        )
    except Exception as e:
        print(f"\n☠ Error during upload: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()