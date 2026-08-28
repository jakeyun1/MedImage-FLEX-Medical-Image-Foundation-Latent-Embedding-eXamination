"""Load and aggregate the original patient-level ODIR-5K annotations."""

from collections import Counter

import numpy as np
import pandas as pd

from scripts.data_audit import require_columns
from scripts.dataset_contracts import get_dataset_contract


ODIR_LABELS = ("N", "D", "G", "C", "A", "H", "M", "O")


def _clean_patient_ids(values):
    ids = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if ids.eq("").any() or ids.str.casefold().eq("nan").any():
        raise ValueError("ODIR patient IDs contain missing or blank values.")
    return ids


def prepare_odir_eye_records(source_df):
    """Expand each patient row into validated left- and right-eye image rows."""
    contract = get_dataset_contract("odir")
    require_columns(
        source_df.columns,
        contract.required_source_columns,
        "ODIR source workbook",
    )
    if source_df.empty:
        raise ValueError("ODIR source workbook contains no patient rows.")

    prepared = source_df.copy()
    prepared["ID"] = _clean_patient_ids(prepared["ID"])
    if prepared["ID"].duplicated().any():
        duplicates = prepared.loc[prepared["ID"].duplicated(), "ID"].head(5).tolist()
        raise ValueError(f"ODIR source workbook has duplicate patient IDs: {duplicates}")

    label_frame = prepared[list(ODIR_LABELS)].apply(pd.to_numeric, errors="raise")
    if label_frame.isna().any().any():
        raise ValueError("ODIR label columns contain missing values.")
    label_frame = label_frame.astype(int)
    observed_values = set(np.unique(label_frame.to_numpy()).tolist())
    if not observed_values.issubset({0, 1}):
        raise ValueError(f"ODIR labels must be binary; observed {sorted(observed_values)}")
    active_label_counts = label_frame.sum(axis=1)
    if active_label_counts.lt(1).any():
        raise ValueError("Every ODIR patient must have at least one positive label.")

    records = []
    for row_index, row in prepared.iterrows():
        diagnosis = label_frame.loc[row_index].astype(int).tolist()
        for eye, filename_column, keyword_column in (
            ("left", "Left-Fundus", "Left-Diagnostic Keywords"),
            ("right", "Right-Fundus", "Right-Diagnostic Keywords"),
        ):
            filename = str(row[filename_column]).strip()
            if not filename or filename.casefold() == "nan":
                raise ValueError(
                    f"ODIR patient {row['ID']} has a missing {eye}-eye filename."
                )
            records.append({
                "ID": row["ID"],
                "filename": filename,
                "eye": eye,
                "Diagnosis": list(diagnosis),
                "Patient Age": row["Patient Age"],
                "Patient Sex": row["Patient Sex"],
                "Diagnostic Keywords": row[keyword_column],
            })

    eye_records = pd.DataFrame.from_records(records)
    if eye_records["filename"].duplicated().any():
        duplicates = eye_records.loc[
            eye_records["filename"].duplicated(), "filename"
        ].head(5).tolist()
        raise ValueError(f"ODIR eye filenames are not unique: {duplicates}")
    if not (eye_records.groupby("ID")["eye"].agg(set) == {"left", "right"}).all():
        raise ValueError("Every ODIR patient must map to one left and one right eye.")

    audit = {
        "protocol": "odir_patient_multilabel_v1",
        "source_patient_rows": int(len(prepared)),
        "patient_groups": int(prepared["ID"].nunique()),
        "eye_image_records": int(len(eye_records)),
        "images_per_patient": 2,
        "label_names": list(ODIR_LABELS),
        "active_label_count_distribution": {
            str(count): int(frequency)
            for count, frequency in sorted(Counter(active_label_counts).items())
        },
        "multilabel_patients": int(active_label_counts.gt(1).sum()),
        "label_counts": {
            label: int(label_frame[label].sum()) for label in ODIR_LABELS
        },
        "uses_age_or_sex_as_model_input": False,
    }
    return eye_records, audit


def pool_odir_patient_embeddings(embeddings, eye_metadata, image_paths, normalize=True):
    """Mean-pool the two correctly matched eye embeddings for each patient."""
    contract = get_dataset_contract("odir")
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(image_paths):
        raise ValueError("ODIR embedding rows must match the extracted image paths.")

    image_ids = [contract.sample_id_from_path(path) for path in image_paths]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("ODIR extracted image IDs are not unique.")
    if eye_metadata["filename"].duplicated().any():
        raise ValueError("ODIR eye metadata contains duplicate filenames.")
    metadata_ids = set(eye_metadata["filename"].astype(str))
    if set(image_ids) != metadata_ids:
        missing = sorted(metadata_ids - set(image_ids))
        extra = sorted(set(image_ids) - metadata_ids)
        raise ValueError(
            "ODIR eye embeddings do not match metadata; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )

    index_by_image_id = {sample_id: index for index, sample_id in enumerate(image_ids)}
    pooled_embeddings = []
    patient_records = []
    patient_ids = []
    for patient_id, rows in eye_metadata.groupby("ID", sort=False):
        if len(rows) != 2 or set(rows["eye"]) != {"left", "right"}:
            raise ValueError(
                f"ODIR patient {patient_id} does not have exactly one left and right eye."
            )
        labels = rows["Diagnosis"].map(tuple)
        if labels.nunique() != 1:
            raise ValueError(f"ODIR patient {patient_id} has inconsistent eye labels.")

        ordered = rows.set_index("eye").loc[["left", "right"]]
        indices = [index_by_image_id[filename] for filename in ordered["filename"]]
        pooled = array[indices].mean(axis=0)
        if normalize:
            norm = float(np.linalg.norm(pooled))
            if not np.isfinite(norm) or norm <= np.finfo(np.float32).tiny:
                raise ValueError(
                    f"ODIR patient {patient_id} has a zero or invalid pooled embedding."
                )
            pooled = pooled / norm

        patient_id = str(patient_id)
        pooled_embeddings.append(pooled)
        patient_ids.append(patient_id)
        patient_records.append({
            "ID": patient_id,
            "Diagnosis": list(labels.iloc[0]),
        })

    patient_embeddings = np.asarray(pooled_embeddings, dtype=np.float32)
    patient_metadata = pd.DataFrame.from_records(patient_records)
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("ODIR pooled patient IDs are not unique.")
    if len(patient_embeddings) != len(patient_metadata):
        raise ValueError("ODIR pooled embeddings and patient metadata are misaligned.")

    audit = {
        "evaluation_unit": "patient",
        "source_eye_embeddings": int(len(array)),
        "evaluation_samples": int(len(patient_embeddings)),
        "aggregation": "l2_normalized_mean_of_left_and_right_eye_embeddings"
        if normalize else "mean_of_left_and_right_eye_embeddings",
        "eyes_per_patient": 2,
        "uses_age_or_sex_as_model_input": False,
    }
    return patient_embeddings, patient_metadata, patient_ids, audit
