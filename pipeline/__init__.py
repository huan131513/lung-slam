"""Endoscopic 3D reconstruction pipeline."""

import sys as _sys

# Guards against the exact bug this project hit in practice: leaving a
# DIFFERENT conda env (e.g. droid-w, from the separate DROID-W project)
# active in a shell, then cd-ing back here and running `python run.py ...`.
# droid_runner.py invokes DROID-SLAM's official demo.py via sys.executable,
# so the wrong env silently runs it with an incompatible droid_backends
# build (droid-w has its own fork-specific one) -- this fails fast instead,
# before any heavier submodule (sam2_runner, viz_live, ...) import can
# produce a confusing unrelated traceback under the wrong env.
if "/envs/droidenv/" not in _sys.executable:
    _sys.exit(
        "\n✗ This must be run with the lung-slam 'droidenv' conda environment.\n"
        f"  Currently running under: {_sys.executable}\n\n"
        "  This almost always means a DIFFERENT conda env (e.g. droid-w) was left\n"
        "  active in this shell. Fix:\n"
        "    conda deactivate   # repeat if nested\n"
        "    conda activate droidenv\n"
        "    python run.py ...\n"
    )
