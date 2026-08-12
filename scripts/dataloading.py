"""
dataloading.py

This file contains the functions for loading the datasets.
"""

import os
import pandas as pd
import kagglehub
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Map to link datasets to their respective paths and their respective CSV data
# FIXME: Change dataset paths as needed
DATASET_MAP = {
    "pad_ufes": {"dataset_path": kagglehub.dataset_download("mahdavi1202/skin-cancer"),
            "CSV_NAMES": ["metadata.csv"]},
    "chexpert": {"dataset_path": kagglehub.dataset_download("ashery/chexpert"),
            "CSV_NAMES": ["train.csv"], "IMAGE_DIRECTORY": "train"},
    "cbis_ddsm": {"dataset_path": kagglehub.dataset_download("debjeetdas/breast-cancer-jpg-image-dataset-of-cbisddsm"),
            "CSV_NAMES": ["calc_case(with_jpg_img).csv", "mass_case(with_jpg_img).csv"]},
    "odir": {"dataset_path": kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k"),
            "CSV_NAMES": ["full_df.csv"],
            "IMAGE_DIRECTORY": f"ODIR-5K{os.sep}Training Images"},
    "ham10000": {"dataset_path": kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"),
            "CSV_NAMES": ["HAM10000_metadata.csv"],
            "IMAGE_DIRECTORY": ["ham10000_images_part_1","ham10000_images_part_2"]}
}

# GeneralDataset class for various datasets for use in pipeline, inherits from PyTorch's Dataset class
class GeneralDataset(Dataset):
    def __init__(self, metadata_df, image_paths, transform = None):
        self.metadata_df = metadata_df
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path

# Custom function for batch loading
def custom_collate_function(batch):
        """
        Prepares a batch for embedding extraction.

        Args:
            batch : A batch of images and image paths from the DataLoader
        
        Returns:
            A Python list of a tuple of images and a tuple of image paths
        """
        return list(zip(*batch))

# Function to effectively load datasets based on their structure
def load_dataset(dataset_name, transform = None, batch_size = 32, shuffle = False, num_workers = 0):
    """
    Loads the desired dataset for use in embedding extraction.

    Args:
        dataset_name : The name of a dataset
        transform : Transform to be applied to each image in the dataset
        batch_size : Number of images to be loaded at a time when extracting embeddings
        shuffle : If True, a random assortment of images will be loaded when extracting embeddings

    Returns:
        dataloader : DataLoader object that contains the image batches for embedding extraction
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
    """
    if dataset_name in DATASET_MAP:
        dataset_path = DATASET_MAP[dataset_name]["dataset_path"]
        CSV_NAMES = DATASET_MAP[dataset_name]["CSV_NAMES"]
        try:
            IMAGE_DIRECTORY = DATASET_MAP[dataset_name]["IMAGE_DIRECTORY"]
        except KeyError:
            pass
    else:
        raise ValueError(f"Dataset \"{dataset_name}\" not recognized.")
    
    image_paths = []
    csv_paths = []

    metadata_df = pd.DataFrame()

    if dataset_name == "cbis_ddsm":
        PATHOLOGIES = ["BENIGN", "MALIGNANT", "BENIGN_WITHOUT_CALLBACK"]

        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if file.endswith(".csv") and file in CSV_NAMES:
                        csv_paths.append(os.path.join(dirpath, file))
                elif file.endswith((".png", ".jpg", ".jpeg")) and not file.startswith("."):
                    image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue
        
        for csv in csv_paths:
            curr_df = pd.read_csv(csv)

            # Ensure multilabel diagnoses (e.g. BENIGN -> mass_BENIGN)
            curr_df["pathology"] = curr_df["abnormality type"] + "_" + curr_df["pathology"]
            metadata_df = pd.concat([metadata_df, curr_df], ignore_index = True)

        image_columns = [
            "jpg_fullMammo_img_path"
        ]

        # Temporary DataFrame used for valid image filtering
        image_df = metadata_df[["pathology"] + image_columns].copy()

        image_df["metadata_index"] = image_df.index

        # Convert the three image columns into one image_path column
        image_df = image_df.melt(
            id_vars = ["metadata_index", "pathology"],
            value_vars = image_columns,
            value_name = "image_path"
        )

        # Remove missing image paths
        image_df = image_df.dropna(
            subset=["image_path"]
        )

        # Split pathology into abnormality type + diagnosis
        #  e.g. mass_BENIGN
        #        -> mass
        #        -> BENIGN
        image_df[
            ["abnormality_type", "diagnosis"]
        ] = (
            image_df["pathology"]
            .str.split("_", n = 1, expand = True)
        )

        # For each image + abnormality type, collect all unique diagnoses.
        diagnoses_by_image = (
            image_df
            .groupby(
                ["image_path", "abnormality_type"]
            )["diagnosis"]
            .agg(set)
        )

        # Find conflicting image + abnormality combinations.
        conflicting_annotations = {
            (image_path, abnormality_type)
            for (
                image_path,
                abnormality_type
            ), diagnoses in diagnoses_by_image.items()
            if len(
                diagnoses.intersection(PATHOLOGIES)
            ) > 1
        }

        # Find the original metadata rows involved in conflicts
        conflict_mask = image_df.apply(
            lambda row: (
                row["image_path"],
                row["abnormality_type"]
            ) in conflicting_annotations,
            axis = 1
        )

        conflicting_metadata_indices = set(
            image_df.loc[
                conflict_mask,
                "metadata_index"
            ]
        )

        # Remove the metadata rows involved in conflicts.
        metadata_df = (
            metadata_df[
                ~metadata_df.index.isin(
                    conflicting_metadata_indices
                )
            ]
            .reset_index(drop = True)
        )

        # Determine which images are no longer represented by any remaining metadata row
        remaining_image_paths = set(
            metadata_df[image_columns]
            .stack()
            .dropna()
        )

        # Remove images that no longer have any valid metadata associated with them.
        image_paths = [
            path
            for path in image_paths
            if any(
                remaining_path in path
                for remaining_path in remaining_image_paths
            )
        ]

        # Make a single, unified `image path` column while keeping all of the original metadata columns
        image_df = metadata_df.melt(
            id_vars = [
                col for col in metadata_df.columns
                if col not in image_columns
            ],
            value_vars = image_columns,
            value_name = "image path"
        )

        # Remove missing image paths
        image_df = image_df.dropna(
            subset = ["image path"]
        )

        # Extract the shared scan identifier (e.g., P_00001_LEFT_CC)
        scan_ids = image_df["image path"].str.extract(r"(P_\d+_(?:LEFT|RIGHT)_(?:CC|MLO))")[0]
        
        # Add a grouping column, falling back to the image path if no match is found
        image_df["scan_id"] = scan_ids.fillna(image_df["image path"])

        # Remove duplicate scan_id + pathology pairs
        image_df = image_df.drop_duplicates(
            subset = ["scan_id", "pathology"]
        )

        # Everything except the structural columns should be preserved as metadata
        metadata_columns = [
            col
            for col in image_df.columns
            if col not in ["image path", "image_type", "pathology", "scan_id"]
        ]

        # Aggregate pathology into a list, while keeping the other metadata columns
        aggregation = {
            "pathology": lambda x: list(dict.fromkeys(x)),
            "image path": "first"  # Pick one valid path representing a scan
        }

        aggregation.update({
            col: "first"
            for col in metadata_columns
        })

        # Group by the extracted identifier to merge Mass and Calcification labels
        metadata_df = (
            image_df
            .groupby(
                "scan_id",
                as_index = False
            )
            .agg(aggregation)
        )

        # Prepare image path column for merging
        metadata_df["image path"] = metadata_df["image path"].apply(lambda path: "/".join(path.split("/")[-2:]))
            
    elif dataset_name == "ham10000":
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if file.endswith(".csv") and file in CSV_NAMES:
                    csv_paths.append(os.path.join(dirpath, file))
                elif file.endswith((".png", ".jpg", ".jpeg")) and not file.startswith("."):
                    for dirname in IMAGE_DIRECTORY:
                        if dirname in dirpath:
                            image_paths.append(os.path.join(dirpath, file))
                            break
            if len(dirnames) != 0:
                continue
        
        for csv in csv_paths:
            curr_df = pd.read_csv(csv)
            metadata_df = pd.concat([metadata_df, curr_df], ignore_index = True)
        
        # Function for concatenating the .jpg extension (due to HAM10000 structure)
        def add_extension(path):
            return path + ".jpg"
        
        metadata_df["image_id"] = metadata_df["image_id"].apply(add_extension)
        
    elif dataset_name == "chexpert":
        LABEL_COLS = ["Cardiomegaly", "Pleural Effusion", "Edema", "Consolidation", "Atelectasis"]
        
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if file.endswith(".csv") and file in CSV_NAMES:
                    csv_paths.append(os.path.join(dirpath, file))
                elif file.endswith((".png", ".jpg", ".jpeg")) and not file.startswith("."):
                    if IMAGE_DIRECTORY in dirpath:
                        image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue
        
        for csv in csv_paths:
            curr_df = pd.read_csv(csv)
            metadata_df = pd.concat([metadata_df, curr_df], ignore_index = True)

        # Assume an uncertain pathology is present, fill empty cells
        metadata_df[LABEL_COLS] = (metadata_df[LABEL_COLS].replace(-1, 1).fillna(0).astype(int))

        # Custom multi-label "Diagnosis" column
        metadata_df["Diagnosis"] = metadata_df[LABEL_COLS].astype(int).values.tolist()

        # Function for retrieving the patient ID for each image
        def get_patient_id(path):
            return path.split("/")[-3]

        # Create the custom grouping column
        metadata_df["patient_id"] = metadata_df["Path"].apply(get_patient_id)

        # Function for retrieving the unique relative paths for each image
        def get_relative_path(path):
            return os.sep.join(path.split("/")[-3:])

        metadata_df["Path"] = metadata_df["Path"].apply(get_relative_path)

        # Image paths inputed into the GeneralDataset are those from the desired .csv files
        desired_paths = set(metadata_df["Path"])
        image_paths = [path for path in image_paths if os.sep.join(path.split(os.sep)[-3:]) in desired_paths]
        
    elif dataset_name == "odir":
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if file.endswith(".csv") and file in CSV_NAMES:
                    csv_paths.append(os.path.join(dirpath, file))
                elif file.endswith((".png", ".jpg", ".jpeg")) and not file.startswith("."):
                    if IMAGE_DIRECTORY in dirpath:
                        image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue

        for csv in csv_paths:
            curr_df = pd.read_csv(csv)
            metadata_df = pd.concat([metadata_df, curr_df], ignore_index = True)
            
        # Image paths inputed into the GeneralDataset are those from the desired .csv files
        images_present = list(metadata_df["filename"])
        image_paths = [path for path in image_paths if os.path.basename(path) in images_present]
    
    else: # For PAD-UFES-20, general datasets
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if file.endswith(".csv") and file in CSV_NAMES:
                    csv_paths.append(os.path.join(dirpath, file))
                elif file.endswith((".png", ".jpg", ".jpeg")) and not file.startswith("."):
                    image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue

        for csv in csv_paths:
            curr_df = pd.read_csv(csv)
            metadata_df = pd.concat([metadata_df, curr_df], ignore_index = True)

    general_dataset = GeneralDataset(metadata_df, image_paths, transform)

    dataloader = DataLoader(general_dataset, batch_size, shuffle, \
            collate_fn = custom_collate_function, num_workers = num_workers)
    dataloader.dataset_name = dataset_name
    
    return dataloader, metadata_df