import sys
import torch
import transformers
import numpy as np
from pathlib import Path
import pkg_resources
from packaging import version

def check_core_libraries():
    """Check versions of core ML libraries."""
    print("\n🤖 Core ML Libraries:")
    print(f"  • PyTorch: {torch.__version__}")
    print(f"  • Transformers: {transformers.__version__}")
    print(f"  • NumPy: {np.__version__}")

def check_requirements():
    """Check installed package versions against requirements.txt."""
    print("\n📋 Package Version Check:")
    
    req_path = Path(__file__).parent.parent / "requirements.txt"
    if not req_path.exists():
        print("  ☠ requirements.txt not found")
        return

    with open(req_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                # Parse package name and version
                package_req = line.strip()
                try:
                    # Get package name and required version
                    name = package_req.split('==')[0]
                    required_version = package_req.split('==')[1]
                    
                    # Get installed version
                    installed_version = pkg_resources.get_distribution(name).version
                    
                    # Compare versions
                    if version.parse(installed_version) == version.parse(required_version):
                        print(f"  ✅ {name}: {installed_version} (matches requirement)")
                    else:
                        print(f"  ⚠️ {name}: installed {installed_version}, required {required_version}")
                except pkg_resources.DistributionNotFound:
                    print(f"  ❌ {name}: not installed")
                except IndexError:
                    # Handle cases where version isn't specified with ==
                    print(f"  ℹ️ {package_req} (version not specified)")

def check_environment():
    """Check and display the software environment configuration."""
    print("\n=== 🔍 Environment Check ===")
    
    # Python Version
    print(f"\n📚 Python Version: {sys.version.split()[0]}")
    
    # Check Core ML Libraries
    check_core_libraries()
    
    # Check Requirements
    check_requirements()

    print("\n=== 🥷  Environment Check Complete  🥷  ===\n")

if __name__ == "__main__":
    check_environment()