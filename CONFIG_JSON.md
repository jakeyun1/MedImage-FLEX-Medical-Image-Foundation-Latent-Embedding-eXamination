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

        "max_samples": Maximum number of samples in the evaluation cohort;
                       null uses all eligible samples
                       (optional, default is 5000)
    },

    "embeddings": {
        "normalize": Flag for normalizing embeddings (optional, default is true),
        "cache": Flag for caching embeddings (optional, default is false)
    },

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

    "embeddings": {
        "cache": true
    },

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
IDs, grouping IDs, label signatures, and outer-fold assignments. Outer and inner
tuning folds are grouped by patient for PAD-UFES, CBIS-DDSM, CheXpert, and ODIR,
and by lesion for HAM10000, whose metadata does not provide patient IDs. Later
model runs with the same configuration validate and reuse that manifest. Every
result records the manifest path and SHA-256 checksum.

## Datasets
- **Chest radiographs**
    - CheXpert: `"chexpert"`
- **Skin lesions**
    - PAD-UFES-20: `"pad_ufes"`
    - HAM10000: `"ham10000"`
- **Mammograms**
    - CBIS-DDSM abnormality crops: `"cbis_ddsm"`
- **Ocular fundi**
    - ODIR-5K: `"odir"`

CBIS-DDSM uses protocol `cbis_ddsm_cropped_v1`: one cropped abnormality image
per pathology row from the configured mass and calcification training metadata.
The loader records any recovered or excluded source rows in the result metadata,
and the protocol name separates its caches and manifests from the previous
full-mammogram workflow.

## Citations
Irvin et al. CheXpert Chest X-rays. Stanford AIMI, 2025. doi:10.71718/y7pj-4v93.​

Pacheco et al. PAD-UFES-20: Skin lesions from smartphones. Mendeley Data, 2020. doi:10.17632/zr7vgbcyr2.1.​

Tschandl et al. The HAM10000 dataset. Harvard Dataverse, 2018. doi:10.7910/DVN/DBW86T.​

Lee et al. CBIS-DDSM: Curated Breast Imaging Subset. TCIA, 2016. doi:10.7937/K9/TCIA.2016.7O02S9CY.​

ODIR-5K: Ocular Disease Intelligent Recognition. Kaggle, 2025. [Online]. [https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k.​](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
