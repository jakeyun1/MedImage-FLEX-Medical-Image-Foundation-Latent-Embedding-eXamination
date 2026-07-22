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
    "cbis_ddsm": {"dataset_path": kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset"),
            "CSV_NAMES": ["dicom_info.csv", "calc_case_description_train_set.csv", "mass_case_description_train_set.csv"]},
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
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if file.endswith(".csv") and file in CSV_NAMES:
                    if file == "dicom_info.csv":
                        mapping_df = pd.read_csv(os.path.join(dirpath, file))
                        uid_to_jpg = list(zip(mapping_df["SeriesInstanceUID"], mapping_df["image_path"]))
                    else:
                        csv_paths.append(os.path.join(dirpath, file))
                elif file.endswith((".png", ".jpg", ".jpeg")) and not file.startswith("."):
                    image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue
        
        for csv in csv_paths:
            curr_df = pd.read_csv(csv)
            metadata_df = pd.concat([metadata_df, curr_df], ignore_index = True)
        
        # Function for translating .dcm paths to .jpg paths
        def map_if_contains(path):
                for uid, jpg_path in uid_to_jpg:
                    if uid in path:
                        return jpg_path
                return path # Return original path if no UID-path relation is found

        # Image paths inputed into the GeneralDataset are those from the desired .csv files
        # Map .dcm to .jpg
        metadata_df["image file path"] = metadata_df["image file path"].apply(map_if_contains)

        # Desired images are in the metadata_df
        images_present = metadata_df["image file path"]

        # Set of desired, unique images with their identifier (`SSUID`/`basename`.jpg)
        images_present = {os.sep.join((x.split("/"))[-2:]) for x in images_present}

        # Full image paths are filtered for those desired
        image_paths = [x for x in image_paths if os.sep.join((x.split(os.sep))[-2:]) in images_present]

        def transform_path(path):
            """
            Transforms a dataset path to match an expected, local path.

            Args:
                path : A path from the CBIS-DDSM dataset

            Returns:
                A localized path used for image identification
            """
            return os.sep.join(path.split("/")[-2:])
        
        # Unique identifier is stored
        metadata_df["image file path"] = metadata_df["image file path"].apply(transform_path)
    
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