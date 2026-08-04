"""
Prepare abnormality-level CBIS-DDSM cropped-image samples.
"""

import os


CBIS_CROP_ID_COL = "cropped image file path"
CBIS_PROTOCOL = "cbis_ddsm_cropped_v1"


def _series_uid(path):
    parts = str(path).strip().replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else None


def _relative_jpeg_path(path):
    parts = str(path).strip().replace("\\", "/").split("/")
    return os.path.join(*parts[-2:])


def prepare_cbis_ddsm_crops(metadata_df, dicom_info_df, image_paths):
    """
    Map each abnormality row to one cropped JPEG and discard unmappable rows.

    The ROI-mask reference is used only as a fallback for the known case where
    the dataset's crop and mask references are swapped.
    """
    descriptions = dicom_info_df["SeriesDescription"].astype(str).str.strip().str.lower()
    crop_rows = dicom_info_df[descriptions.eq("cropped images")]

    crop_counts = crop_rows.groupby("SeriesInstanceUID")["image_path"].nunique()
    ambiguous_uids = crop_counts[crop_counts != 1]
    if not ambiguous_uids.empty:
        raise ValueError(
            "CBIS-DDSM crop series must map to exactly one JPEG; ambiguous UIDs include "
            f"{ambiguous_uids.index[:5].tolist()}."
        )

    crop_by_uid = (
        crop_rows.drop_duplicates("SeriesInstanceUID")
        .set_index("SeriesInstanceUID")["image_path"]
        .map(_relative_jpeg_path)
        .to_dict()
    )
    image_by_id = {_relative_jpeg_path(path): path for path in image_paths}
    if len(image_by_id) != len(image_paths):
        raise ValueError("CBIS-DDSM JPEG paths contain duplicate identifiers.")

    mapped_ids = []
    mapping_sources = []
    for _, row in metadata_df.iterrows():
        primary = crop_by_uid.get(_series_uid(row[CBIS_CROP_ID_COL]))
        fallback = crop_by_uid.get(_series_uid(row["ROI mask file path"]))

        candidates = list(dict.fromkeys(path for path in (primary, fallback) if path))
        if len(candidates) > 1:
            raise ValueError(
                "CBIS-DDSM row maps to multiple cropped JPEGs: "
                f"{candidates}."
            )

        mapped_id = candidates[0] if candidates else None
        if mapped_id not in image_by_id:
            mapped_id = None
        mapped_ids.append(mapped_id)
        mapping_sources.append(
            None if mapped_id is None
            else "primary" if primary == mapped_id
            else "roi_fallback"
        )

    prepared = metadata_df.copy()
    prepared[CBIS_CROP_ID_COL] = mapped_ids
    prepared["_crop_mapping_source"] = mapping_sources
    excluded = prepared[prepared[CBIS_CROP_ID_COL].isna()]
    prepared = prepared.dropna(subset=[CBIS_CROP_ID_COL]).copy()
    recovered = prepared[prepared["_crop_mapping_source"] == "roi_fallback"]

    if prepared[CBIS_CROP_ID_COL].duplicated().any():
        raise ValueError("CBIS-DDSM cropped-image identifiers must be unique.")

    selected_paths = [image_by_id[path] for path in prepared[CBIS_CROP_ID_COL]]
    audit = {
        "protocol": CBIS_PROTOCOL,
        "image_unit": "abnormality_crop",
        "input_rows": int(len(metadata_df)),
        "usable_crops": int(len(prepared)),
        "roi_fallback_rows": int((prepared["_crop_mapping_source"] == "roi_fallback").sum()),
        "roi_fallback_samples": [
            {
                "patient_id": str(row["patient_id"]),
                "laterality": str(row["left or right breast"]),
                "view": str(row["image view"]),
                "abnormality_id": int(row["abnormality id"]),
                "pathology": str(row["pathology"]),
                "crop_id": str(row[CBIS_CROP_ID_COL]),
            }
            for _, row in recovered.iterrows()
        ],
        "excluded_rows": int(len(excluded)),
        "excluded_samples": [
            {
                "patient_id": str(row["patient_id"]),
                "laterality": str(row["left or right breast"]),
                "view": str(row["image view"]),
                "abnormality_id": int(row["abnormality id"]),
                "pathology": str(row["pathology"]),
            }
            for _, row in excluded.iterrows()
        ],
    }
    prepared = prepared.drop(columns=["_crop_mapping_source"]).reset_index(drop=True)
    return prepared, selected_paths, audit
