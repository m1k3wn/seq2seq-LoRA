from transformers import T5Tokenizer
from pathlib import Path
import shutil
import os
import time

# Get root directory where script is located
ROOT_DIR = Path(__file__).parent

# Download spiece.model to a temporary location first
print("Downloading base tokenizer...")
temp_dir = ROOT_DIR / "temp_tokenizer"
temp_dir.mkdir(exist_ok=True)

base_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", local_files_only=False)
base_tokenizer.save_pretrained(temp_dir)

# Verify spiece.model exists and get its details
spiece_path = temp_dir / "spiece.model"
if spiece_path.exists():
    print(f"Found spiece.model in temp dir: {spiece_path}")
    print(f"Size: {spiece_path.stat().st_size} bytes")
else:
    raise FileNotFoundError("spiece.model not found in downloaded tokenizer!")

# Copy to one model directory first as a test
test_model_dir = ROOT_DIR / "merged_models" / "nidra_v1_merged"
target_path = test_model_dir / "spiece.model"

print(f"\nCopying spiece.model to {test_model_dir}")
shutil.copy2(spiece_path, target_path)

# Check immediately after copy
print("\nChecking immediately after copy:")
if target_path.exists():
    print(f"File exists at {target_path}")
    print(f"Size: {target_path.stat().st_size} bytes")
    print(f"Permissions: {oct(target_path.stat().st_mode)[-3:]}")
else:
    print("File does not exist immediately after copy")

# Wait a second and check again
time.sleep(1)
print("\nChecking after 1 second:")
if target_path.exists():
    print(f"File still exists at {target_path}")
else:
    print("File no longer exists after 1 second")

# List all files in directory
print("\nAll files in directory:")
for file in test_model_dir.iterdir():
    print(f"- {file.name}: {file.stat().st_size} bytes")