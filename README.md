# MedImage-FLEX: Medical Image Foundation Latent Embedding eXamination

https://parrarodrigu.github.io/medimage-flex/

## Setup
1. Prior to setup, it is recommended to set up a separate environment using the correct Python version (3.11.15)
```
# Example using conda
conda create -n testbench python=3.11.15
conda activate testbench
```
2. Clone the GitHub repo
3. Navigate into to [`environment`](environment/) and run `python setup.py` to download dependencies
```
cd environment
python setup.py
cd ..
```
## Running the Testbench
1. Choose your model
2. Add its `model_id` and loading logic to [`scripts/models.py`](scripts/models.py)
3. Create a model configuration JSON file with respect to the format in `CONFIG_JSON.md`
4. Run the testbench
```
# Example
python main.py --config ./model_config.json --num-workers 2
```
5. MedImage-FLEX offers default model logic as a convenience for three popular frameworks: PyTorch, HuggingFace, and TensorFlow. If a model requires specific logic not handled by default, edit [`scripts/model_interface.py`](scripts/model_interface.py), [`scripts/custom_transforms.py`](scripts/custom_transforms.py) as needed.

**NOTE: Any personal access tokens or keys for models must be loaded locally**