# Configuration JSON Format
```
# Format
{
    "model_id": Unique model ID, path, or link used to instantiate the model,

    "output_dir": Custom results output directory
                  (optional, default is results/),

    "dataset": {
        "datasets": [
            list,
            of,
            dataset,
            names
        ],

        "batch_size": Desired batch size
                      (optional, default is 32),

        "shuffle": Flag for shuffling images while computing embeddings
                   (optional, default is false),

        "max_samples": Approximate number of evaluation samples in the cohort;
                       complete patient/lesion groups are selected, so the actual
                       count may be slightly above or below this target;
                       ODIR uses patients while the other datasets use images;
                       null uses all eligible samples
                       (optional, default is 5000),

        "chexpert_uncertainty_policy": CheXpert uncertain-label policy. One of
                       "finding_specific" (primary), "u_zeros", or "u_ones"
                       (optional, default is "finding_specific")
    },

    "normalize_embeddings": Flag for normalizing embeddings (optional, default is true),

    "cache_embeddings": Flag for caching embeddings (optional, default is false),

    "random_baseline": {
        "enabled": Run the complete benchmark on dimension-matched random Gaussian
                   embeddings (optional, default is false),
        "repeats": Number of random embedding seeds (optional, default is 20),
        "seed": First random embedding seed; subsequent runs increment it by one
                (optional, default is 42),
        "overwrite": Recompute existing per-seed results instead of resuming
                     (optional, default is false)
    },

    "label_prior_baseline": {
        "enabled": Run majority-class and stratified-random classification
                   controls (optional, default is false)
    },

    "permuted_baseline": {
        "enabled": Run the complete benchmark after permuting genuine embeddings
                   among the selected samples (optional, default is false),
        "repeats": Number of permutation seeds (optional, default is 20),
        "seed": First permutation seed; subsequent runs increment it by one
                (optional, default is 42),
        "overwrite": Recompute existing per-seed results instead of resuming
                     (optional, default is false)
    },

    "reproducibility": {
        "manifest_dir": Shared directory for cohort and fold manifests
                        (optional, default is manifests/),
        "sample_seed": Seed for stratified cohort selection
                       (optional, default is 42),
        "outer_folds": Number of outer cross-validation folds
                       (optional, default is 5),
        "fold_seed": Seed used to create outer folds
                     (optional, default is 42),
        "evaluation_seed": Seed for inner folds, Optuna, and estimators
                           (optional, default is 42)
    }
}

# Sample
{
    "model_id": "microsoft/rad-dino",

    "output_dir": "custom_output_folder",

    "dataset": {
        "datasets": [
            "pad_ufes",
            "cbis_ddsm",
            "ham10000"
        ],

        "batch_size": 20,

        "shuffle": false,
        "max_samples": 5000
    },

    "cache_embeddings": true,

    "random_baseline": {
        "enabled": true,
        "repeats": 20,
        "seed": 42,
        "overwrite": false
    },

    "label_prior_baseline": {
        "enabled": true
    },

    "permuted_baseline": {
        "enabled": true,
        "repeats": 20,
        "seed": 42,
        "overwrite": false
    },

    "reproducibility": {
        "manifest_dir": "manifests",
        "sample_seed": 42,
        "outer_folds": 5,
        "fold_seed": 42,
        "evaluation_seed": 42
    }
}
```

When enabled, baseline results are saved beneath
`<run_folder>/random_baseline/<dataset>/`. Each seed has a full benchmark JSON,
and `summary.json` reports the mean, sample standard deviation, and empirical
95% percentile interval across seeds. Random embeddings are regenerated from
their recorded seeds rather than stored.

Label-prior results are saved beneath `<run_folder>/label_prior_baseline/`.
These controls use only label frequencies from each outer training fold.
Permuted-embedding results are saved beneath
`<run_folder>/permuted_baseline/<dataset>/`; only embeddings belonging to the
selected cohort are permuted with no fixed points, while image paths and labels
remain fixed.

The first run for a dataset creates a cohort manifest containing its exact sample
IDs, grouping IDs, label signatures, and outer-fold assignments. Cohort selection,
outer folds, and inner tuning folds preserve complete patient groups for PAD-UFES,
CBIS-DDSM, CheXpert, and ODIR, and complete lesion groups for HAM10000, whose
metadata does not provide patient IDs. Later model runs with the same configuration
validate and reuse that manifest. Every result records the manifest path and SHA-256
checksum, along with the requested and actual cohort sizes.

MLP, KNN, and logistic-regression hyperparameters are all selected by maximizing
macro F1 within the inner cross-validation folds. This gives each class or finding
equal weight during model selection instead of allowing frequent labels to dominate.

Every dataset result also contains two documentation blocks:

- `data_audit.fingerprints` includes a traversal-independent SHA-256 of retained
  sample IDs and a SHA-256 of canonical sample/group/processed-label records. These
  are lightweight semantic fingerprints and deliberately do not hash image bytes.
- `run_provenance` records a canonical SHA-256 of `config_used.json`'s content, the
  Git commit/branch and separate tracked-change/untracked-file states when available,
  Python and platform details, and relevant installed package versions. Tracked
  changes mean the commit alone is not a complete description of the executed code;
  untracked files are reported separately because they may be non-code artifacts.

The ordered sample-ID hashes in `embedding_info` serve a different purpose: they
document the exact row order of the embedding arrays and are expected to change if
extraction order changes.

The primary CheXpert policy maps uncertain Cardiomegaly and Consolidation labels
to absent and uncertain Atelectasis, Edema, and Pleural Effusion labels to present.
Unmentioned labels map to absent. The alternative `u_zeros` and `u_ones` settings
are intended for prespecified sensitivity analyses.

ODIR is evaluated as the original patient-level multilabel task. The left- and
right-eye embeddings for each of the 3,500 labeled patients are mean-pooled and
L2-normalized. Age and sex are recorded in the source audit but are not model
inputs in this image-representation benchmark.

## Datasets
- **Chest radiographs**
    - CheXpert: `"chexpert"`
- **Skin lesions**
    - PAD-UFES-20: `"pad_ufes"`
    - HAM10000: `"ham10000"`
- **Mammograms**
    - CBIS-DDSM: `"cbis_ddsm"`
- **Ocular fundi**
    - ODIR-5K: `"odir"`

## Citations
Irvin et al. CheXpert Chest X-rays. Stanford AIMI, 2025. doi:10.71718/y7pj-4v93.​

Pacheco et al. PAD-UFES-20: Skin lesions from smartphones. Mendeley Data, 2020. doi:10.17632/zr7vgbcyr2.1.​

Tschandl et al. The HAM10000 dataset. Harvard Dataverse, 2018. doi:10.7910/DVN/DBW86T.​

Breast Cancer JPG Image Dataset of CBIS-DDSM. Kaggle, 2024. [Online]. [https://www.kaggle.com/datasets/debjeetdas/breast-cancer-jpg-image-dataset-of-cbisddsm.​](https://www.kaggle.com/datasets/debjeetdas/breast-cancer-jpg-image-dataset-of-cbisddsm)

ODIR-5K: Ocular Disease Intelligent Recognition. Kaggle, 2025. [Online]. [https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k.​](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
