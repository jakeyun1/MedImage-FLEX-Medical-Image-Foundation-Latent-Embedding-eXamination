"""
setup.py

Downloads the datasets and the testbench dependencies.
"""

import os
import sys
import subprocess

# Make the current directory (for the subprocess) relative to the testbench program
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

python = sys.executable

# Download testbench dependencies
subprocess.run([python, "-m", "pip", "install", "-r", f".{os.sep}dependencies.txt"], check = True)

# Download deep learning frameworks
subprocess.run([python, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch", "torchvision"], check = True)
subprocess.run([python, "-m", "pip", "install", "tensorflow-cpu"], check = True)

# Import after downloading dependencies
import kagglehub

# Download the datasets
# PAD-UFES-20
kagglehub.dataset_download("mahdavi1202/skin-cancer")

# CheXpert
kagglehub.dataset_download("ashery/chexpert")

# CBIS-DDSM
kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")

# ODIR-5K
kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")

# HAM10000
kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("\nTestbench setup done.")