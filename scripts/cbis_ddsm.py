"""Prepare one validated multilabel sample per CBIS-DDSM mammogram scan."""

from collections import Counter
import hashlib
import os

import pandas as pd


SOURCE_PATH_COLUMN = "jpg_fullMammo_img_path"
CBIS_LABELS = (
    "mass_BENIGN",
    "mass_MALIGNANT",
    "calcification_BENIGN",
    "calcification_MALIGNANT",
)
REQUIRED_COLUMNS = (
    "patient_id",
    "left or right breast",
    "image view",
    "abnormality type",
    "pathology",
    SOURCE_PATH_COLUMN,
)
PATHOLOGY_MAP = {
    "BENIGN": "BENIGN",
    "BENIGN_WITHOUT_CALLBACK": "BENIGN",
    "MALIGNANT": "MALIGNANT",
}


def _sample_id(path):
    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"CBIS-DDSM image path has fewer than two components: {path}")
    return "/".join(parts[-2:])


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonblank(frame, columns):
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"CBIS-DDSM column contains missing values: {column}")
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"CBIS-DDSM column contains blank values: {column}")


def prepare_cbis_ddsm_full_mammograms(metadata_df, image_paths):
    """Aggregate annotations and select one verified file for each mammogram scan."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(metadata_df.columns))
    if missing_columns:
        raise ValueError(f"CBIS-DDSM metadata is missing columns: {missing_columns}")
    if metadata_df.empty:
        raise ValueError("CBIS-DDSM metadata contains no rows.")

    prepared = metadata_df.copy()
    _require_nonblank(prepared, REQUIRED_COLUMNS)
    prepared["patient_id"] = prepared["patient_id"].astype(str).str.strip()
    prepared["left or right breast"] = (
        prepared["left or right breast"].astype(str).str.strip().str.upper()
    )
    prepared["image view"] = prepared["image view"].astype(str).str.strip().str.upper()
    prepared["abnormality type"] = (
        prepared["abnormality type"].astype(str).str.strip().str.lower()
    )
    if not set(prepared["abnormality type"]).issubset({"mass", "calcification"}):
        unexpected = sorted(
            set(prepared["abnormality type"]) - {"mass", "calcification"}
        )
        raise ValueError(f"CBIS-DDSM has unexpected abnormality types: {unexpected}")

    source_pathology = prepared["pathology"].astype(str).str.strip().str.upper()
    unexpected_pathology = sorted(set(source_pathology) - set(PATHOLOGY_MAP))
    if unexpected_pathology:
        raise ValueError(f"CBIS-DDSM has unexpected pathology values: {unexpected_pathology}")
    prepared["_pathology"] = source_pathology.map(PATHOLOGY_MAP)
    prepared["_label"] = prepared["abnormality type"] + "_" + prepared["_pathology"]
    if not set(prepared["_label"]).issubset(CBIS_LABELS):
        raise ValueError("CBIS-DDSM produced labels outside the declared ontology.")

    prepared["_scan_id"] = (
        prepared["patient_id"]
        + "_"
        + prepared["left or right breast"]
        + "_"
        + prepared["image view"]
    )
    path_scan_ids = prepared[SOURCE_PATH_COLUMN].astype(str).str.extract(
        r"(P_\d+_(?:LEFT|RIGHT)_(?:CC|MLO))"
    )[0]
    mismatch = path_scan_ids.isna() | path_scan_ids.ne(prepared["_scan_id"])
    if mismatch.any():
        examples = prepared.loc[mismatch, ["_scan_id", SOURCE_PATH_COLUMN]].head(5)
        raise ValueError(
            "CBIS-DDSM metadata columns do not match image-path scan IDs: "
            f"{examples.to_dict(orient='records')}"
        )

    prepared["_sample_id"] = prepared[SOURCE_PATH_COLUMN].map(_sample_id)
    required_sample_ids = set(prepared["_sample_id"])
    image_by_sample_id = {}
    for image_path in image_paths:
        sample_id = _sample_id(image_path)
        if sample_id not in required_sample_ids:
            continue
        if sample_id in image_by_sample_id:
            raise ValueError(f"CBIS-DDSM image ID maps to multiple files: {sample_id}")
        image_by_sample_id[sample_id] = image_path

    missing_images = sorted(required_sample_ids - set(image_by_sample_id))
    if missing_images:
        raise ValueError(
            f"CBIS-DDSM metadata references missing full mammograms: {missing_images[:5]}"
        )

    partition = prepared[SOURCE_PATH_COLUMN].astype(str).str.extract(
        r"/(?:Calc|Mass)_(Training|Test)_"
    )[0]
    if partition.isna().any():
        raise ValueError("CBIS-DDSM could not determine source partition from image paths.")
    prepared["_source_partition"] = partition

    records = []
    selected_image_paths = []
    duplicate_aliases_collapsed = 0
    for scan_id, scan_rows in prepared.groupby("_scan_id", sort=True):
        sample_ids = sorted(set(scan_rows["_sample_id"]))
        candidate_paths = [image_by_sample_id[sample_id] for sample_id in sample_ids]
        if len(candidate_paths) > 1:
            hashes = {_file_sha256(path) for path in candidate_paths}
            if len(hashes) != 1:
                raise ValueError(
                    f"CBIS-DDSM scan maps to different image contents: {scan_id}"
                )
            duplicate_aliases_collapsed += len(candidate_paths) - 1

        patient_ids = set(scan_rows["patient_id"])
        if len(patient_ids) != 1:
            raise ValueError(f"CBIS-DDSM scan maps to multiple patients: {scan_id}")
        selected_sample_id = sample_ids[0]
        selected_image_paths.append(image_by_sample_id[selected_sample_id])
        records.append({
            "scan_id": scan_id,
            "image path": selected_sample_id,
            "pathology": sorted(set(scan_rows["_label"])),
            "patient_id": next(iter(patient_ids)),
            "left or right breast": scan_rows["left or right breast"].iloc[0],
            "image view": scan_rows["image view"].iloc[0],
            "source_partitions": sorted(set(scan_rows["_source_partition"])),
        })

    result = pd.DataFrame.from_records(records)
    label_counts = Counter(label for labels in result["pathology"] for label in labels)
    audit = {
        "protocol": "cbis_ddsm_full_mammogram_multilabel_v1",
        "source_abnormality_rows": int(len(prepared)),
        "source_full_mammogram_paths": int(prepared["_sample_id"].nunique()),
        "canonical_scans": int(len(result)),
        "patient_groups": int(result["patient_id"].nunique()),
        "duplicate_image_aliases_collapsed": int(duplicate_aliases_collapsed),
        "benign_without_callback_rows_collapsed": int(
            source_pathology.eq("BENIGN_WITHOUT_CALLBACK").sum()
        ),
        "mixed_source_partition_scans": int(
            result["source_partitions"].map(len).gt(1).sum()
        ),
        "label_counts": dict(sorted(label_counts.items())),
    }
    return result, selected_image_paths, audit
