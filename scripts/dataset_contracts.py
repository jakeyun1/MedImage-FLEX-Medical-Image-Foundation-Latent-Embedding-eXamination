"""Stable dataset metadata and sample-identity rules.

Runtime loading and evaluation use these contracts as their shared source of truth.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DatasetContract:
    """Fields required to identify and validate one supported dataset."""

    name: str
    dataset_handle: str
    metadata_filenames: Tuple[str, ...]
    id_column: str
    label_column: str
    group_column: str
    label_type: str
    required_source_columns: Tuple[str, ...]
    sample_id_path_parts: int = 1
    image_directory_names: Tuple[str, ...] = ()
    image_id_column: str = ""
    label_names: Tuple[str, ...] = ()
    evaluation_unit: str = "image"
    exclusion_policy_filename: str = ""

    def __post_init__(self):
        if self.label_type not in {"multiclass", "multilabel"}:
            raise ValueError(f"Unsupported label type: {self.label_type}")
        if self.sample_id_path_parts < 1:
            raise ValueError("sample_id_path_parts must be at least 1.")
        if not self.metadata_filenames:
            raise ValueError("At least one metadata filename is required.")
        if self.label_type == "multilabel" and not self.label_names:
            raise ValueError("Multilabel datasets must declare label names.")
        if not self.evaluation_unit:
            raise ValueError("An evaluation unit is required.")

    @property
    def load_id_column(self) -> str:
        """ID used for the one-to-one metadata-to-image loading audit."""
        return self.image_id_column or self.id_column

    def sample_id_from_path(self, path) -> str:
        """Return a platform-independent sample ID from a local image path."""
        normalized = str(path).replace("\\", "/").rstrip("/")
        parts = [part for part in normalized.split("/") if part]
        if len(parts) < self.sample_id_path_parts:
            raise ValueError(
                f"Path has fewer than {self.sample_id_path_parts} components: {path}"
            )
        return "/".join(parts[-self.sample_id_path_parts:])

    def path_is_in_image_directory(self, path) -> bool:
        """Check configured image directories by path component, ignoring case."""
        if not self.image_directory_names:
            return True
        path_parts = {
            part.casefold()
            for part in str(path).replace("\\", "/").split("/")
            if part
        }
        return any(
            directory_name.casefold() in path_parts
            for directory_name in self.image_directory_names
        )


DATASET_CONTRACTS: Dict[str, DatasetContract] = {
    "pad_ufes": DatasetContract(
        name="pad_ufes",
        dataset_handle="mahdavi1202/skin-cancer/versions/1",
        metadata_filenames=("metadata.csv",),
        id_column="img_id",
        label_column="diagnostic",
        group_column="patient_id",
        label_type="multiclass",
        required_source_columns=("img_id", "diagnostic", "patient_id"),
        exclusion_policy_filename="pad_ufes_exclusions.csv",
    ),
    "chexpert": DatasetContract(
        name="chexpert",
        dataset_handle="ashery/chexpert/versions/1",
        metadata_filenames=("train.csv",),
        id_column="Path",
        label_column="Diagnosis",
        group_column="patient_id",
        label_type="multilabel",
        label_names=(
            "Cardiomegaly",
            "Pleural Effusion",
            "Edema",
            "Consolidation",
            "Atelectasis",
        ),
        required_source_columns=(
            "Path",
            "Cardiomegaly",
            "Pleural Effusion",
            "Edema",
            "Consolidation",
            "Atelectasis",
            "Frontal/Lateral",
        ),
        sample_id_path_parts=3,
        image_directory_names=("train",),
        evaluation_unit="radiograph",
        exclusion_policy_filename="chexpert_exclusions.csv",
    ),
    "cbis_ddsm": DatasetContract(
        name="cbis_ddsm",
        dataset_handle=(
            "debjeetdas/breast-cancer-jpg-image-dataset-of-cbisddsm/versions/1"
        ),
        metadata_filenames=(
            "calc_case(with_jpg_img).csv",
            "mass_case(with_jpg_img).csv",
        ),
        id_column="image path",
        label_column="pathology",
        group_column="patient_id",
        label_type="multilabel",
        label_names=(
            "mass_BENIGN",
            "mass_MALIGNANT",
            "calcification_BENIGN",
            "calcification_MALIGNANT",
        ),
        required_source_columns=(
            "abnormality type",
            "pathology",
            "jpg_fullMammo_img_path",
            "patient_id",
            "left or right breast",
            "image view",
        ),
        sample_id_path_parts=2,
        evaluation_unit="full_mammogram",
    ),
    "odir": DatasetContract(
        name="odir",
        dataset_handle="andrewmvd/ocular-disease-recognition-odir5k/versions/2",
        metadata_filenames=("data.xlsx",),
        id_column="ID",
        image_id_column="filename",
        label_column="Diagnosis",
        group_column="ID",
        label_type="multilabel",
        label_names=("N", "D", "G", "C", "A", "H", "M", "O"),
        required_source_columns=(
            "ID",
            "Patient Age",
            "Patient Sex",
            "Left-Fundus",
            "Right-Fundus",
            "Left-Diagnostic Keywords",
            "Right-Diagnostic Keywords",
            "N",
            "D",
            "G",
            "C",
            "A",
            "H",
            "M",
            "O",
        ),
        image_directory_names=("Training Images",),
        evaluation_unit="patient",
    ),
    "ham10000": DatasetContract(
        name="ham10000",
        dataset_handle="kmader/skin-cancer-mnist-ham10000/versions/2",
        metadata_filenames=("HAM10000_metadata.csv",),
        id_column="image_id",
        label_column="dx",
        group_column="lesion_id",
        label_type="multiclass",
        required_source_columns=("image_id", "dx", "lesion_id"),
        image_directory_names=(
            "ham10000_images_part_1",
            "ham10000_images_part_2",
        ),
        exclusion_policy_filename="ham10000_exclusions.csv",
    ),
}


def get_dataset_contract(dataset_name: str) -> DatasetContract:
    """Return the contract for a supported dataset."""
    try:
        return DATASET_CONTRACTS[dataset_name]
    except KeyError as exc:
        raise ValueError(f'Unknown dataset: "{dataset_name}"') from exc
