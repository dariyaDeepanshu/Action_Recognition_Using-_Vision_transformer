# =============================================================================
# DOWNLOAD PRE‑TRAINED CHECKPOINTS FROM GOOGLE DRIVE
# =============================================================================
# This script downloads model checkpoints that have been uploaded to Google
# Drive because they exceed GitHub's file size limit (100 MB per file).
# The checkpoints are stored in a shared Drive folder; you must replace the
# placeholder file IDs with the actual IDs from the shared links.
#
# Dependencies:
#   pip install gdown
#
# Usage:
#   python scripts/download_checkpoints.py
#
# How to obtain a file ID:
#   1. Share the file with "Anyone with the link".
#   2. Copy the link; it looks like:
#        https://drive.google.com/file/d/FILE_ID/view?usp=sharing
#   3. The FILE_ID is the long string between /d/ and /view.
# =============================================================================

import os
import sys
from typing import Dict


# -----------------------------------------------------------------------------
# Configuration: mapping from destination path to Google Drive file ID.
# Update these IDs with the actual values from your shared Drive folder.
# -----------------------------------------------------------------------------
CHECKPOINT_FILES: Dict[str, str] = {
    "checkpoints/videomae_best.pt":    "184xIN_fUneZ-mZQZRTphi8XSAsb0jW3u",
    "checkpoints/timesformer_best.pt": "1nzR621gMPaeGZ8vjOMfoTaXGEbiTSluR",
    "checkpoints/detr_best.pt":        "187GXMedxPj6X3q7HTmsfm1180LEZD_ri",
    # Multi‑seed checkpoints (optional, used in `multi_seed_evaluation`)
    "checkpoints/seed_42.pt":          "1AyLekGwFflF71rwgnKYXQtLFa8uPAWb3",
    "checkpoints/seed_123.pt":         "1udf-0qpHyCJWVhLKz4_Su_yQyC7H7CQW",
    "checkpoints/seed_456.pt":         "1jLyxqloglrC45WZZxxV1B_b3UwYxwsBq",
}


def check_gdown() -> None:
    """
    Verify that the `gdown` package is installed. Exit with an error message
    if it is missing.
    """
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[ERROR] 'gdown' is not installed. Please install it using:")
        print("        pip install gdown")
        sys.exit(1)


def download_file(file_id: str, dest_path: str) -> None:
    """
    Download a single file from Google Drive using its file ID.

    Args:
        file_id   : Google Drive file ID (extracted from the share link).
        dest_path : Local path where the file will be saved.
    """
    import gdown

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Skip downloading if the file already exists
    if os.path.exists(dest_path):
        print(f"[SKIP]  {dest_path} already exists.")
        return

    # Construct the download URL and start the download
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[DOWN]  {dest_path} ...")
    gdown.download(url, dest_path, quiet=False)


def main() -> None:
    """
    Main entry point: iterate over the CHECKPOINT_FILES dictionary and download
    each file that has a valid file ID (not the placeholder). Skip entries with
    placeholder IDs and print a warning.
    """
    # Check that gdown is available
    check_gdown()

    # Placeholder value (should not appear in real usage)
    PLACEHOLDER = "YOUR_GDRIVE_FILE_ID_HERE"

    # Identify entries that still have the placeholder ID
    missing = [k for k, v in CHECKPOINT_FILES.items() if v == PLACEHOLDER]
    if missing:
        print("[WARN] The following entries still have placeholder IDs and will be skipped:")
        for m in missing:
            print(f"       {m}")
        print("       Edit CHECKPOINT_FILES in this script with the real Drive file IDs.\n")

    # Download each file with a valid ID
    for dest_path, file_id in CHECKPOINT_FILES.items():
        if file_id == PLACEHOLDER:
            continue
        download_file(file_id, dest_path)

    print("\nDone. All available checkpoints downloaded to the 'checkpoints/' directory.")


if __name__ == "__main__":
    main()