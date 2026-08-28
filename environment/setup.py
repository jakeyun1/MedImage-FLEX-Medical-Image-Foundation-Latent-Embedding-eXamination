"""
setup.py

Downloads the datasets and the testbench dependencies.
"""

import os
import platform
import sys
import subprocess

# Make the current directory (for the subprocess) relative to the testbench program
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

python = sys.executable

# Download testbench dependencies
subprocess.run([python, "-m", "pip", "install", "-r", f".{os.sep}dependencies.txt"], check = True)

# Download deep learning frameworks. macOS wheels are published on PyPI rather
# than the framework-specific CPU indexes used on Linux and Windows.
torch_command = [
    python, "-m", "pip", "install", "torch==2.8.0", "torchvision==0.23.0"
]
if platform.system() != "Darwin":
    torch_command[4:4] = ["--index-url", "https://download.pytorch.org/whl/cpu"]
subprocess.run(torch_command, check = True)

tensorflow_package = (
    "tensorflow==2.18.1"
    if platform.system() == "Darwin"
    else "tensorflow-cpu==2.18.1"
)
subprocess.run([python, "-m", "pip", "install", tensorflow_package], check = True)

# Import after downloading dependencies
import kagglehub

# Download the datasets
# PAD-UFES-20
kagglehub.dataset_download("mahdavi1202/skin-cancer/versions/1")

# CheXpert
kagglehub.dataset_download("ashery/chexpert/versions/1")

# CBIS-DDSM
kagglehub.dataset_download("debjeetdas/breast-cancer-jpg-image-dataset-of-cbisddsm/versions/1")

# ODIR-5K
kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k/versions/2")

# HAM10000
kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000/versions/2")

print("\nTestbench setup done.")
