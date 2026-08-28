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
    read_dataset_protocol,
    read_exclusion_policy,
    require_columns,
    resolve_required_files,
)
from scripts.dataset_contracts import get_dataset_contract
from scripts.odir import prepare_odir_eye_records
from scripts.provenance import build_dataset_fingerprint


CHEXPERT_LABELS = (
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Consolidation",
    "Atelectasis",
)
CHEXPERT_FINDING_SPECIFIC_POLICY = {
    "Cardiomegaly": 0,
    "Pleural Effusion": 1,
    "Edema": 1,
    "Consolidation": 0,
    "Atelectasis": 1,
}


def _read_metadata(csv_paths, contract):
    resolved_paths = resolve_required_files(
        csv_paths,
        contract.metadata_filenames,
        f'{contract.name} metadata discovery',
    )
    frames = []
    for metadata_path in resolved_paths:
        extension = os.path.splitext(metadata_path)[1].casefold()
        if extension == ".csv":
            frame = pd.read_csv(metadata_path)
        elif extension in {".xlsx", ".xls"}:
            frame = pd.read_excel(metadata_path)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path}")
        require_columns(
            frame.columns,
            contract.required_source_columns,
            f'{contract.name} metadata file "{metadata_path}"',
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
    exclusion_audit,
):
    require_columns(
        metadata_df.columns,
        (contract.load_id_column, contract.label_column, contract.group_column),
        f"{contract.name} processed metadata",
    )
    if metadata_df.empty:
        raise ValueError(f"{contract.name} metadata contains no rows.")
    if not image_paths:
        raise ValueError(f"{contract.name} contains no matched image files.")

    metadata_ids = metadata_df[contract.load_id_column].astype(str).tolist()
    image_ids = [contract.sample_id_from_path(path) for path in image_paths]
    audit = IdentityAudit.from_ids(metadata_ids, image_ids).validate()
    expected, protocol_manifest = read_dataset_protocol(contract.name)
    observed = {
        "dataset_handle": contract.dataset_handle,
        "source_metadata_rows": int(source_metadata_rows),
        "discovered_image_files": int(discovered_image_files),
        "retained_image_files": int(audit.image_rows),
        "evaluation_unit": contract.evaluation_unit,
        "label_type": contract.label_type,
    }
    mismatches = {
        key: {"expected": expected[key], "observed": value}
        for key, value in observed.items()
        if expected[key] != value
    }
    if mismatches:
        raise ValueError(
            f"{contract.name} does not match its audited dataset protocol: {mismatches}"
        )
    summary = audit.to_dict()
    summary.update({
        "dataset": contract.name,
        "dataset_handle": contract.dataset_handle,
        "evaluation_unit": contract.evaluation_unit,
        "dataset_root": os.path.abspath(dataset_path),
        "metadata_files": [os.path.abspath(path) for path in csv_paths],
        "source_metadata_rows": int(source_metadata_rows),
        "discovered_image_files": int(discovered_image_files),
        "unretained_discovered_image_files": int(
            discovered_image_files - audit.image_rows
        ),
        "policy_excluded_image_files": int(
            exclusion_audit["excluded_samples"] if exclusion_audit else 0
        ),
        "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
        "fingerprints": build_dataset_fingerprint(
            metadata_ids,
            metadata_df[contract.group_column].astype(str).tolist(),
            metadata_df[contract.label_column].tolist(),
        ),
        "exclusion_policy": exclusion_audit,
        "protocol_manifest": protocol_manifest,
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


def _apply_exclusion_policy(contract, metadata_df, image_paths):
    policy, policy_audit = read_exclusion_policy(
        contract.exclusion_policy_filename
    )
    if not policy:
        return metadata_df, image_paths, None

    excluded_ids = set(policy)
    metadata_ids = metadata_df[contract.load_id_column].astype(str)
    image_ids = [contract.sample_id_from_path(path) for path in image_paths]
    missing_metadata = sorted(excluded_ids - set(metadata_ids))
    missing_images = sorted(excluded_ids - set(image_ids))
    if missing_metadata or missing_images:
        raise ValueError(
            f"{contract.name} exclusion policy does not match the pinned dataset; "
            f"missing metadata IDs={missing_metadata[:5]}, "
            f"missing image IDs={missing_images[:5]}."
        )
    if metadata_ids[metadata_ids.isin(excluded_ids)].duplicated().any():
        raise ValueError(
            f"{contract.name} exclusion policy IDs are not unique in metadata."
        )

    retained_metadata = metadata_df.loc[~metadata_ids.isin(excluded_ids)].copy()
    retained_paths = [
        path for path, sample_id in zip(image_paths, image_ids)
        if sample_id not in excluded_ids
    ]
    if len(metadata_df) - len(retained_metadata) != len(excluded_ids):
        raise ValueError(f"{contract.name} did not exclude every declared metadata row.")
    if len(image_paths) - len(retained_paths) != len(excluded_ids):
        raise ValueError(f"{contract.name} did not exclude every declared image file.")
    return retained_metadata, retained_paths, policy_audit


def _apply_chexpert_uncertainty_policy(metadata_df, policy_name):
    valid_policies = {"finding_specific", "u_zeros", "u_ones"}
    if policy_name not in valid_policies:
        raise ValueError(
            "CheXpert uncertainty policy must be one of "
            f"{sorted(valid_policies)}; found {policy_name!r}."
        )

    raw = metadata_df[list(CHEXPERT_LABELS)].apply(pd.to_numeric, errors="raise")
    unexpected = sorted(
        set(raw.stack().dropna().unique().tolist()) - {-1.0, 0.0, 1.0}
    )
    if unexpected:
        raise ValueError(f"CheXpert has unexpected label values: {unexpected}")

    if policy_name == "finding_specific":
        replacements = CHEXPERT_FINDING_SPECIFIC_POLICY
    else:
        replacements = {
            label: int(policy_name == "u_ones") for label in CHEXPERT_LABELS
        }
    processed = raw.copy()
    for label in CHEXPERT_LABELS:
        processed[label] = processed[label].replace(-1, replacements[label])
    processed = processed.fillna(0).astype(int)

    result = metadata_df.copy()
    result[list(CHEXPERT_LABELS)] = processed
    result["Diagnosis"] = processed.to_numpy().tolist()
    audit = {
        "uncertainty_policy": policy_name,
        "uncertain_value_mapping": dict(replacements),
        "missing_value_mapping": 0,
        "source_raw_uncertain_counts": {
            label: int(raw[label].eq(-1).sum()) for label in CHEXPERT_LABELS
        },
        "source_raw_missing_counts": {
            label: int(raw[label].isna().sum()) for label in CHEXPERT_LABELS
        },
        "source_view_counts": {
            str(view): int(count)
            for view, count in result["Frontal/Lateral"].value_counts().items()
        },
        "sample_unit": "radiograph",
    }
    return result, audit

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
def load_dataset(dataset_name, transform = None, batch_size = 32, shuffle = False,
                 num_workers = 0, chexpert_uncertainty_policy = "finding_specific"):
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
    contract = get_dataset_contract(dataset_name)
    dataset_path = kagglehub.dataset_download(contract.dataset_handle)
    CSV_NAMES = list(contract.metadata_filenames)
    
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

        metadata_df, dataset_specific_audit = _apply_chexpert_uncertainty_policy(
            metadata_df, chexpert_uncertainty_policy
        )

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

        source_df, csv_paths = _read_metadata(csv_paths, contract)
        source_metadata_rows = len(source_df)
        discovered_image_files = len(image_paths)
        metadata_df, dataset_specific_audit = prepare_odir_eye_records(source_df)
            
        # Image paths inputed into the GeneralDataset are those from the desired .csv files
        images_present = set(metadata_df["filename"])
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

    metadata_df, image_paths, exclusion_audit = _apply_exclusion_policy(
        contract, metadata_df, image_paths
    )
    if dataset_name == "chexpert":
        dataset_specific_audit["retained_view_counts"] = {
            str(view): int(count)
            for view, count in metadata_df["Frontal/Lateral"].value_counts().items()
        }

    audit_summary = _audit_loaded_dataset(
        contract,
        metadata_df,
        image_paths,
        csv_paths,
        dataset_path,
        source_metadata_rows,
        discovered_image_files,
        exclusion_audit,
    )
    if dataset_specific_audit is not None:
        audit_summary["dataset_specific"] = dataset_specific_audit

    general_dataset = GeneralDataset(metadata_df, image_paths, transform)

    dataloader = DataLoader(general_dataset, batch_size, shuffle, \
            collate_fn = custom_collate_function, num_workers = num_workers)
    dataloader.dataset_name = dataset_name
    dataloader.data_audit = audit_summary
    
    return dataloader, metadata_df
