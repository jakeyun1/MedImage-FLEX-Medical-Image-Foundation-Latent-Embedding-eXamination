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

## Dataset protocol

Dataset downloads are pinned to explicit Kaggle versions. Exact-content duplicate
exclusions are stored in `dataset_audits/`, validated against metadata and image
IDs at load time, and identified by SHA-256 in each result.

PAD-UFES uses patient-grouped folds. HAM10000 uses lesion-grouped folds because
patient identifiers are unavailable. CheXpert retains frontal and lateral views as
separate radiograph samples but keeps every image from one patient in one fold; its
primary uncertainty policy is documented in `CONFIG_JSON.md`. CBIS-DDSM evaluates
the third-party JPEG full-mammogram derivative as an abnormality-enriched dataset,
with source partitions pooled for patient-grouped cross-validation.

ODIR uses the original `data.xlsx` patient annotations, not the derived one-hot
`full_df.csv`. Each evaluation sample is one patient represented by the normalized
mean of the left- and right-eye embeddings and an eight-element multilabel target.

Each dataset result includes lightweight provenance metadata. The data audit records
a traversal-independent SHA-256 over retained sample IDs and another over canonical
sample/group/processed-label records. These fingerprints document dataset semantics;
they do not hash image bytes or establish clinical label correctness. The existing
ordered sample-ID hashes remain separate because they document embedding row order.
Results also record a canonical configuration SHA-256, Git commit and separate
tracked/untracked-worktree state, Python runtime, platform, and relevant package
versions.
