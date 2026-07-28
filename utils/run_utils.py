# =============================================================================
# SHARED HELPERS FOR NAMING AND ORGANISING SCRIPT OUTPUTS
# =============================================================================
# Every script that produces result artefacts (plots, metrics, JSON summaries)
# should write them under its own timestamped subdirectory of outputs/, e.g.
#   outputs/evaluate_classifier/videomae_20260727_193015/metrics.json
#   outputs/evaluate_classifier/videomae_20260727_193015/confusion_matrix.png
#
# This keeps checkpoints/ and logs/ (which are re-used across runs by path,
# e.g. checkpoints/videomae_best.pt) separate from run outputs (which should
# never silently overwrite a previous run's results).
# =============================================================================

import os
from datetime import datetime

OUTPUTS_ROOT = "outputs"


def make_run_dir(script_name: str, *tags: str, base: str = OUTPUTS_ROOT) -> str:
    """
    Create and return a fresh, timestamped output directory for one run of a
    script: outputs/<script_name>/<tag1>_<tag2>_..._<timestamp>/

    All artefacts produced by that run (plots, JSON metrics, etc.) should be
    written inside the returned directory using plain, descriptive filenames
    (e.g. "metrics.json") -- the directory name already records what was run
    and when, so results from different runs never collide or overwrite each
    other.

    Args:
        script_name: Name of the calling script (e.g. "train_classifier").
        *tags      : Extra identifying tags, in order (e.g. model name,
                     experiment/ablation name). Falsy tags (None, "") are
                     skipped.
        base       : Root output directory (default "outputs").

    Returns:
        Path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "_".join(str(t) for t in tags if t)
    run_name = f"{label}_{timestamp}" if label else timestamp
    run_dir = os.path.join(base, script_name, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
