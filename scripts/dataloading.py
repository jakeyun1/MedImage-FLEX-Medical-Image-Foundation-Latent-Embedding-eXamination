"""
dataloading.py

This file contains the functions for loading the datasets.
"""

import os
import pandas as pd
import kagglehub
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from scripts.cbis_ddsm import prepare_cbis_ddsm_full_mammograms
from scripts.data_audit import (
    IdentityAudit,
    ordered_ids_sha256,
    require_columns,
    resolve_required_files,
)
from scripts.dataset_contracts import DATASET_CONTRACTS, get_dataset_contract

# Map to link datasets to their respective paths and their respective CSV data
# FIXME: Change dataset paths as needed
DATASET_MAP = {
    name: {
        "dataset_path": kagglehub.dataset_download(contract.dataset_handle),
        "CSV_NAMES": list(contract.metadata_filenames),
    }
    for name, contract in DATASET_CONTRACTS.items()
}


def _read_metadata(csv_paths, contract):
    resolved_paths = resolve_required_files(
        csv_paths,
        contract.metadata_filenames,
        f'{contract.name} metadata discovery',
    )
    frames = []
    for csv_path in resolved_paths:
        frame = pd.read_csv(csv_path)
        require_columns(
            frame.columns,
            contract.required_source_columns,
            f'{contract.name} metadata file "{csv_path}"',
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), resolved_paths


def _is_metadata_file(filename, expected_filenames):
    return filename.casefold() in {
        expected_filename.casefold()
        for expected_filename in expected_filenames
    }


def _is_image_file(filename):
    return (
        not filename.startswith(".")
        and os.path.splitext(filename)[1].casefold() in {".png", ".jpg", ".jpeg"}
    )


def _audit_loaded_dataset(
    contract,
    metadata_df,
    image_paths,
    csv_paths,
    dataset_path,
    source_metadata_rows,
    discovered_image_files,
):
    require_columns(
        metadata_df.columns,
        (contract.id_column, contract.label_column, contract.group_column),
        f"{contract.name} processed metadata",
    )
    if metadata_df.empty:
        raise ValueError(f"{contract.name} metadata contains no rows.")
    if not image_paths:
        raise ValueError(f"{contract.name} contains no matched image files.")

    metadata_ids = metadata_df[contract.id_column].astype(str).tolist()
    image_ids = [contract.sample_id_from_path(path) for path in image_paths]
    audit = IdentityAudit.from_ids(metadata_ids, image_ids).validate()
    summary = audit.to_dict()
    summary.update({
        "dataset": contract.name,
        "dataset_root": os.path.abspath(dataset_path),
        "metadata_files": [os.path.abspath(path) for path in csv_paths],
        "source_metadata_rows": int(source_metadata_rows),
        "discovered_image_files": int(discovered_image_files),
        "excluded_image_files": int(discovered_image_files - audit.image_rows),
        "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
    })
    print(
        f"Data audit [{contract.name}]: "
        f"source_metadata_rows={source_metadata_rows}, "
        f"processed_metadata_rows={audit.metadata_rows}, "
        f"discovered_image_files={discovered_image_files}, "
        f"matched_image_files={audit.image_rows}, "
        f"matched_ids={audit.matched_unique_ids}, "
        f"ordered_ids_sha256={summary['ordered_image_ids_sha256']}"
    )
    return summary

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
        contract = get_dataset_contract(dataset_name)
        dataset_path = DATASET_MAP[dataset_name]["dataset_path"]
        CSV_NAMES = DATASET_MAP[dataset_name]["CSV_NAMES"]
    else:
        raise ValueError(f"Dataset \"{dataset_name}\" not recognized.")
    
    image_paths = []
    csv_paths = []

    metadata_df = pd.DataFrame()
    dataset_specific_audit = None

    if dataset_name == "cbis_ddsm":
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if _is_metadata_file(file, CSV_NAMES):
                    csv_paths.append(os.path.join(dirpath, file))
                elif _is_image_file(file):
                    image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue
        
        metadata_df, csv_paths = _read_metadata(csv_paths, contract)
        source_metadata_rows = len(metadata_df)
        discovered_image_files = len(image_paths)

        metadata_df, image_paths, dataset_specific_audit = (
            prepare_cbis_ddsm_full_mammograms(metadata_df, image_paths)
        )
            
    elif dataset_name == "ham10000":
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if _is_metadata_file(file, CSV_NAMES):
                    csv_paths.append(os.path.join(dirpath, file))
                elif _is_image_file(file):
                    if contract.path_is_in_image_directory(dirpath):
                        image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue
        
        metadata_df, csv_paths = _read_metadata(csv_paths, contract)
        source_metadata_rows = len(metadata_df)
        discovered_image_files = len(image_paths)
        
        # Function for concatenating the .jpg extension (due to HAM10000 structure)
        def add_extension(path):
            return path + ".jpg"
        
        metadata_df["image_id"] = metadata_df["image_id"].apply(add_extension)
        
    elif dataset_name == "chexpert":
        LABEL_COLS = ["Cardiomegaly", "Pleural Effusion", "Edema", "Consolidation", "Atelectasis"]
        
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if _is_metadata_file(file, CSV_NAMES):
                    csv_paths.append(os.path.join(dirpath, file))
                elif _is_image_file(file):
                    if contract.path_is_in_image_directory(dirpath):
                        image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue
        
        metadata_df, csv_paths = _read_metadata(csv_paths, contract)
        source_metadata_rows = len(metadata_df)
        discovered_image_files = len(image_paths)

        # Assume an uncertain pathology is present, fill empty cells
        metadata_df[LABEL_COLS] = (metadata_df[LABEL_COLS].replace(-1, 1).fillna(0).astype(int))

        # Custom multi-label "Diagnosis" column
        metadata_df["Diagnosis"] = metadata_df[LABEL_COLS].astype(int).values.tolist()

        # Create canonical sample and patient IDs from the relative image path
        metadata_df["Path"] = metadata_df["Path"].apply(contract.sample_id_from_path)
        metadata_df["patient_id"] = metadata_df["Path"].str.split("/").str[0]

        # Image paths inputed into the GeneralDataset are those from the desired .csv files
        desired_paths = set(metadata_df["Path"])
        image_paths = [
            path for path in image_paths
            if contract.sample_id_from_path(path) in desired_paths
        ]
        
    elif dataset_name == "odir":
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if _is_metadata_file(file, CSV_NAMES):
                    csv_paths.append(os.path.join(dirpath, file))
                elif _is_image_file(file):
                    if contract.path_is_in_image_directory(dirpath):
                        image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue

        metadata_df, csv_paths = _read_metadata(csv_paths, contract)
        source_metadata_rows = len(metadata_df)
        discovered_image_files = len(image_paths)
            
        # Image paths inputed into the GeneralDataset are those from the desired .csv files
        images_present = list(metadata_df["filename"])
        image_paths = [
            path for path in image_paths
            if contract.sample_id_from_path(path) in images_present
        ]
    
    else: # For PAD-UFES-20, general datasets
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if _is_metadata_file(file, CSV_NAMES):
                    csv_paths.append(os.path.join(dirpath, file))
                elif _is_image_file(file):
                    image_paths.append(os.path.join(dirpath, file))
            if len(dirnames) != 0:
                continue

        metadata_df, csv_paths = _read_metadata(csv_paths, contract)
        source_metadata_rows = len(metadata_df)
        discovered_image_files = len(image_paths)

    audit_summary = _audit_loaded_dataset(
        contract,
        metadata_df,
        image_paths,
        csv_paths,
        dataset_path,
        source_metadata_rows,
        discovered_image_files,
    )
    if dataset_specific_audit is not None:
        audit_summary["dataset_specific"] = dataset_specific_audit

    general_dataset = GeneralDataset(metadata_df, image_paths, transform)

    dataloader = DataLoader(general_dataset, batch_size, shuffle, \
            collate_fn = custom_collate_function, num_workers = num_workers)
    dataloader.dataset_name = dataset_name
    dataloader.data_audit = audit_summary
    
    return dataloader, metadata_df
