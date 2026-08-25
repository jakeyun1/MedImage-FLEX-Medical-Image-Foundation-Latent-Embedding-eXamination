"""Dependency-free validation primitives for dataset loading audits."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Sequence, Tuple


class AuditValidationError(ValueError):
    """Raised when a dataset identity audit violates its declared policy."""


def require_columns(observed: Iterable[str], required: Iterable[str], context: str):
    """Fail with a stable, readable error when required columns are absent."""
    observed_set = set(observed)
    missing = sorted(set(required) - observed_set)
    if missing:
        raise AuditValidationError(
            f"{context} is missing required columns: {missing}"
        )


def resolve_required_files(
    discovered_paths: Iterable[str],
    expected_filenames: Iterable[str],
    context: str,
) -> Tuple[str, ...]:
    """Resolve exactly one copy of each required file, in declared order."""
    paths_by_name = {}
    for path in discovered_paths:
        filename = str(path).replace("\\", "/").rsplit("/", 1)[-1].casefold()
        paths_by_name.setdefault(filename, []).append(str(path))

    resolved = []
    errors = []
    for expected_filename in expected_filenames:
        matches = sorted(paths_by_name.get(expected_filename.casefold(), []))
        if not matches:
            errors.append(f"missing {expected_filename}")
        elif len(matches) > 1:
            errors.append(f"multiple copies of {expected_filename}: {matches}")
        else:
            resolved.append(matches[0])
    if errors:
        raise AuditValidationError(f"{context}: {'; '.join(errors)}")
    return tuple(resolved)


def ordered_ids_sha256(sample_ids: Sequence[str]) -> str:
    """Hash an ordered sample-ID sequence without ambiguous concatenation."""
    payload = json.dumps(
        [str(sample_id) for sample_id in sample_ids],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _duplicates(values: Sequence[str]) -> Tuple[str, ...]:
    counts = Counter(str(value) for value in values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


@dataclass(frozen=True)
class IdentityAudit:
    """Cardinality and identity checks for a metadata-to-image join."""

    metadata_rows: int
    image_rows: int
    metadata_unique_ids: int
    image_unique_ids: int
    matched_unique_ids: int
    metadata_only_ids: Tuple[str, ...]
    image_only_ids: Tuple[str, ...]
    duplicate_metadata_ids: Tuple[str, ...]
    duplicate_image_ids: Tuple[str, ...]

    @classmethod
    def from_ids(cls, metadata_ids: Sequence[str], image_ids: Sequence[str]):
        metadata_values = [str(value) for value in metadata_ids]
        image_values = [str(value) for value in image_ids]
        metadata_set = set(metadata_values)
        image_set = set(image_values)
        return cls(
            metadata_rows=len(metadata_values),
            image_rows=len(image_values),
            metadata_unique_ids=len(metadata_set),
            image_unique_ids=len(image_set),
            matched_unique_ids=len(metadata_set & image_set),
            metadata_only_ids=tuple(sorted(metadata_set - image_set)),
            image_only_ids=tuple(sorted(image_set - metadata_set)),
            duplicate_metadata_ids=_duplicates(metadata_values),
            duplicate_image_ids=_duplicates(image_values),
        )

    def validate(
        self,
        *,
        allow_metadata_only: bool = False,
        allow_image_only: bool = False,
    ):
        errors = []
        if self.duplicate_metadata_ids:
            errors.append(f"duplicate metadata IDs: {list(self.duplicate_metadata_ids)}")
        if self.duplicate_image_ids:
            errors.append(f"duplicate image IDs: {list(self.duplicate_image_ids)}")
        if self.metadata_only_ids and not allow_metadata_only:
            errors.append(f"metadata IDs without images: {list(self.metadata_only_ids)}")
        if self.image_only_ids and not allow_image_only:
            errors.append(f"image IDs without metadata: {list(self.image_only_ids)}")
        if errors:
            raise AuditValidationError("; ".join(errors))
        return self

    def to_dict(self):
        result = asdict(self)
        result["metadata_only_count"] = len(self.metadata_only_ids)
        result["image_only_count"] = len(self.image_only_ids)
        result["duplicate_metadata_count"] = len(self.duplicate_metadata_ids)
        result["duplicate_image_count"] = len(self.duplicate_image_ids)
        return result
