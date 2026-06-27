# Configuration JSON Format #
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

        "shuffle": Flag for shuffling images while
                   computing embeddings
                   (optional, default is false)
    }
}

# Sample
{
    "model_id": "microsoft/rad-dino"

    "output_dir": "custom_output_folder",

    "dataset": {
        "datasets": [
            "pad_ufes",
            "cbis_ddsm",
            "ham10000"
        ],

        "batch_size": 20,

        "shuffle": false
    }
}
```

## Datasets ##
- **Chest radiographs**
    - CheXpert: `"chexpert"`
- **Skin lesions**
    - PAD-UFES-20: `"pad_ufes"`
    - HAM10000: `"ham10000"`
- **Mammograms**
    - CBIS-DDSM: `"cbis_ddsm"`
- **Ocular fundi**
    - ODIR-5K: `"odir"`

## Citations ##
Irvin et al. CheXpert Chest X-rays. Stanford AIMI, 2025. doi:10.71718/y7pj-4v93.​

Pacheco et al. PAD-UFES-20: Skin lesions from smartphones. Mendeley Data, 2020. doi:10.17632/zr7vgbcyr2.1.​

Tschandl et al. The HAM10000 dataset. Harvard Dataverse, 2018. doi:10.7910/DVN/DBW86T.​

Lee et al. CBIS-DDSM: Curated Breast Imaging Subset. TCIA, 2016. doi:10.7937/K9/TCIA.2016.7O02S9CY.​

ODIR-5K: Ocular Disease Intelligent Recognition. Kaggle, 2025. [Online]. [https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k.​](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)