# GitHub upload checklist

This directory is the local release candidate for the public repository.

Before committing, confirm:

1. Run `python verify_release.py` and keep the message `v7 neutral release verification passed`.
2. Run `python heldout_gate/verify_heldout_gate.py` and keep the message `heldout_gate verification passed`.
3. Copy the contents of this directory into the local Git repository while preserving the repository's `.git` directory.
4. Review the GitHub Desktop change list. Do not add raw EEG, Word drafts, local logs, `__pycache__`, `.pyc`, temporary files, or private paths.
5. Commit on a review branch named `submission-v1` with a message such as `Prepare JMBE submission release`.
6. Push the branch, inspect the online files, and merge only after the README, frozen predictions, `heldout_gate/`, and manifest files are correct.
7. Create a release tag only after the merge. Use the tag in the Zenodo archive and in the final citation metadata.

The canonical primary prediction is
`frozen/predictions_recording_route_a_v3.csv` with SHA-256
`cba069dd3837e9b3894dc2649aa6c8a9ac80812de996b2dc51cc0aa056f5c5cd`.
The `heldout_gate/` directory is a supplementary post-hoc audit and does not
replace that canonical file.
