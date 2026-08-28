# Dataset exclusion policies

These files record sample-level exclusions discovered by the one-time exact-content
audit of the pinned dataset versions. Runtime loading validates that every listed ID
exists in both metadata and image files, applies the exclusions, and records the
policy file's SHA-256 digest in the result audit.

Conflicting exact duplicates are all removed. Cross-patient CheXpert duplicates are
all removed. For same-label duplicates within one patient or lesion, one
lexicographically first canonical sample is retained and the redundant copies listed
here are removed. No images or clinical metadata are stored in this repository.
