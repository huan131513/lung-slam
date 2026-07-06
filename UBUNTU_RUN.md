# Ubuntu Quickstart — Direct DROID-SLAM Path

Step-by-step instructions to reproduce a single-clip trajectory reconstruction
on Ubuntu + RTX 3090, using the direct path (no filter, no SAM 2, no `--mask_dir`).

Verified on: Ubuntu 20.04, NVIDIA driver 550 (CUDA 11.6 system `nvcc`),
PyTorch 1.13.1+cu116, Python 3.10.

---

## 0. Assumptions

- You already have the repo cloned at `~/lung-slam` (or clone it now).
- You have Conda / Miniconda installed (used here only to get a Python ≥3.9
  interpreter — system Python 3.8 on Ubuntu 20.04 is too old for current
  PyTorch wheels — everything else is installed with plain `pip`).
- You have an NVIDIA driver + CUDA runtime installed on the host.
- Your video file is somewhere on the host filesystem.

---

## 1. Pull latest code

```bash
cd ~/lung-slam
git pull origin main
```

Verify you have the new files:
```bash
ls pipeline/preprocess.py scripts/preprocess_for_droid.py
python run.py --help | grep -E "preprocess|all-droid"
```
Expected: both files exist and `preprocess` + `all-droid` appear in the CLI help.

---

## 2. Install DROID-SLAM (once per machine)

> **Do not use DROID-SLAM's `environment.yaml` / `conda env create`.** That
> path is now marked deprecated upstream, and on this kind of multi-channel
> old spec (`rusty1s`, `open3d-admin`) conda's solver can run for 20+ minutes
> and then get OOM-killed by the kernel without a clear error — it happened
> twice while writing this doc. Upstream's current README instead recommends
> plain `pip` into a venv (tested by them up to torch 2.7); we use a minimal
> conda env only to get Python 3.10, then do everything else with `pip`.

```bash
cd ~
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM

conda create -n droidenv python=3.10 -y
conda activate droidenv
```

**Pick a PyTorch build whose CUDA *major* version matches your system's
`nvcc --version`** (check with `nvcc --version` and `nvidia-smi`) — building
the CUDA extensions below fails if they mismatch. Example for system CUDA 11.6:

```bash
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu116
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# must print "1.13.1+cu116 True" — if False, the wheel's CUDA build is too
# new/old for your driver (check https://pytorch.org for other cuXXX tags)
```

Now the rest of DROID-SLAM's own install steps, plus two fixes this repo needed:

```bash
pip install -r requirements.txt
pip install moderngl moderngl-window   # demo.py imports this unconditionally now

# thirdparty/lietorch builds fine against the torch just installed:
pip install --no-build-isolation thirdparty/lietorch

# thirdparty/pytorch_scatter's bundled source uses C++17 (std::optional) but
# this repo's build flags are C++14 -> compile error. Use a prebuilt wheel
# matching your torch/cuda tag instead (find yours at data.pyg.org/whl):
pip install torch-scatter==2.1.1+pt113cu116 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# torch==1.13 was built against numpy 1.x; newer numpy/opencv break the
# lietorch/torch_scatter C extensions with an "_ARRAY_API not found" warning
# (and can hard-crash) — pin both down:
pip install "numpy<2" "opencv-python<4.10"

python setup.py install                # builds droid_backends, ~1-3 min
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh # downloads droid.pth into repo root
```

Sanity check the extensions actually built:
```bash
python -c "import torch, lietorch, torch_scatter, droid_backends; print('ok')"
```

Export the path so this repo's runner can find it:
```bash
echo 'export DROID_SLAM_ROOT=$HOME/DROID-SLAM' >> ~/.bashrc
source ~/.bashrc
echo $DROID_SLAM_ROOT               # sanity check
ls $DROID_SLAM_ROOT/droid.pth       # weight must exist
```

---

## 3. Install this repo's Python deps

Inside the same `droidenv` conda env:

```bash
cd ~/lung-slam
pip install -r requirements.txt
sudo apt-get install -y ffmpeg      # if not already present
```

Sanity check:
```bash
python run.py check
```
Should print Python / CUDA / ffmpeg / OpenCV versions with no errors. A
`sam2 NOT installed` warning is expected and fine for this direct path —
SAM 2 is only needed for the advanced Path B (tool masking).

---

## 4. Prepare the test clip

The 10-second sample clip is committed to the repo at
[`data/7348331-618-628.mp4`](data/7348331-618-628.mp4) (~10 MB, 1920×1080 @ 60 fps).
After `git pull` in Step 1, it is already on disk — nothing else to download.

Verify:
```bash
cd ~/lung-slam
ls -lh data/7348331-618-628.mp4
ffprobe -v error -show_entries stream=width,height,r_frame_rate,duration \
        data/7348331-618-628.mp4
```

Cut out the 5-10s subclip (which has the largest camera motion of the source):
```bash
ffmpeg -y -ss 5 -t 5 \
    -i data/7348331-618-628.mp4 \
    -c copy data/clip_5to10.mp4
```

For a fair A / B comparison, also cut 0-5s:
```bash
ffmpeg -y -ss 0 -t 5 \
    -i data/7348331-618-628.mp4 \
    -c copy data/clip_0to5.mp4
```

> **If you want to use your own video instead**, drop it into `data/` (any
> filename) and adjust the paths in Sections 5-8 accordingly. Other `.mp4`
> files remain gitignored — only `7348331-618-628.mp4` has an explicit
> exception in `.gitignore`.

---

## 5. Run the direct DROID-SLAM path

### Primary run — 5-10s at fps=4

```bash
python run.py all-droid \
    --video data/clip_5to10.mp4 \
    --out ./out/clip_5to10 \
    --fps 4 \
    --distortion-margin 0.15 \
    --assumed-fov-deg 90
```

Why `--fps 4`: the source is 60 fps but the average optical flow is only
~1.1 px/frame. DROID-SLAM's frontend keyframe threshold is 16 px, so we
resample to 4 fps to boost per-frame flow to ~17 px — the minimum needed
for keyframes to be selected at all.

### Control run — 0-5s at fps=4 (for comparison)

```bash
python run.py all-droid \
    --video data/clip_0to5.mp4 \
    --out ./out/clip_0to5 \
    --fps 4 \
    --distortion-margin 0.15 \
    --assumed-fov-deg 90
```

---

## 6. Inspect outputs

Each `out/clip_*/` directory should contain:

```
out/clip_5to10/
├── preview.png         # crop overlay (yellow=raw FOV, red=trimmed, green=final crop)
├── calib.txt           # "fx fy cx cy"
├── frames/             # ~20 cropped PNGs (5s × 4fps)
├── metadata.json
├── droid/              # DROID-SLAM output
│   ├── reconstruction.pth  # raw torch.save bundle written by demo.py
│   ├── poses.npy           # (N, 7) camera poses [tx ty tz qx qy qz qw],
│   │                       #   WORLD frame — already inverted from
│   │                       #   DROID-SLAM's internal world-to-camera storage
│   ├── disps.npy           # per-keyframe inverse depth
│   ├── tstamps.npy         # keyframe indices into frames/
│   ├── images.npy          # keyframe images (as seen by DROID-SLAM)
│   ├── intrinsics.npy      # per-keyframe camera intrinsics
│   └── points.ply          # colored world-frame point cloud
│                           #   (droid_backends.iproj + depth_filter)
└── viz/                # trajectory.json + trajectory.png (2D projections)
```

Sanity checks:

| Check | Command | Pass criterion |
|---|---|---|
| Preview crop is correct | open `preview.png` | green box is inside the tissue view, no black border |
| Number of frames | `ls out/clip_5to10/frames \| wc -l` | ~20 |
| DROID converged | `python -c "import numpy as np; p=np.load('out/clip_5to10/droid/poses.npy'); print(p.shape)"` | shape starts with a number > 10 |
| Trajectory shape | open `out/clip_5to10/viz/trajectory.png` | should look like a short arc, not a jittery zigzag |
| Interactive check | `python run.py viz-sync --out ./out/clip_5to10` | video frame + point cloud + camera frustums, step with ←→, see [`viz_sync.py`](pipeline/viz_sync.py) |

---

## 7. Compare A/B and interpret

Open both `viz/` outputs side-by-side. Expected outcomes:

| Result | Meaning |
|---|---|
| Both trajectories jittery, similar length | Pulsation dominates in both segments — this video isn't usable for trajectory |
| 5-10s trajectory clearly longer / smoother than 0-5s | 22% flow increase carried through — the pipeline works, but scale is unreliable |
| Either run crashes | Report the traceback; usually means <12 keyframes were built (increase `--fps`, or extend the clip) |

**Important:** whichever you get, do **NOT** trust the metric scale of a
monocular reconstruction from this kind of clip. Use it only to validate that
the pipeline runs end-to-end. See the "known limitations" notes below.

---

## 8. Troubleshooting

### `DROID_SLAM_ROOT not set or not found`
```bash
export DROID_SLAM_ROOT=$HOME/DROID-SLAM
ls $DROID_SLAM_ROOT/demo.py           # must exist
```

### `failed to detect circular FOV`
The FOV threshold is too high for a dim clip. Retry with:
```bash
python run.py preprocess ... --fov-threshold 8
```

### DROID-SLAM crashes with "not enough frames for initialization"
Too few frames after downsampling. Options:
- Raise `--fps` (e.g. from 4 to 6) — trades baseline for frame count
- Use a longer subclip (`ffmpeg -ss 0 -t 10`)

### `ffmpeg: command not found`
```bash
sudo apt-get install -y ffmpeg
```

### CUDA out of memory
Downscale further:
```bash
python run.py all-droid ... --width 960
```

### `conda env create -f environment.yaml` hangs or the machine grinds to a halt
Don't use it — see the warning at the top of Section 2. Kill it
(`pkill -9 -f "conda env create"`) and follow the pip-based steps instead.

### `ModuleNotFoundError: No module named 'droid'` when running the `droid` stage
`demo.py` does `sys.path.append('droid_slam')` (a relative path) and must be
run with `DROID_SLAM_ROOT` as its working directory. This repo's
`pipeline/droid_runner.py` already sets `cwd` correctly — if you see this,
you're likely on a stale checkout; `git pull`.

### `ModuleNotFoundError: No module named 'moderngl'`
```bash
pip install moderngl moderngl-window
```

### `error: 'std::optional' has not been declared` while building `pytorch_scatter`
Don't build it from `thirdparty/pytorch_scatter` — install the prebuilt wheel
instead (see Section 2): `pip install torch-scatter==<ver>+pt<torch>cu<cuda> -f https://data.pyg.org/whl/torch-<ver>+cu<cuda>.html`.

### `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`
`torch`/`lietorch` here were built against NumPy 1.x. Run
`pip install "numpy<2" "opencv-python<4.10"`.

---

## 9. Known limitations of this test clip

Baseline analysis (measured on Mac before shipping):

| Metric | Value | Note |
|---|---|---|
| Source fps | 60 | 1920×1080 |
| Mean optical flow (0-5s) | 0.93 px/frame @ 60fps | very small |
| Mean optical flow (5-10s) | 1.13 px/frame @ 60fps | +22%, still small |
| Dominant motion frequencies | 0.6 / 0.8 / 1.4 Hz | 36–84 BPM = **cardiac / respiratory pulsation** |
| Blurry frames | 33% (lap_var<200) | motion blur is significant |

**Bottom line:** the recording is dominated by tissue pulsation, not camera
translation. Monocular SLAM cannot separate these. Treat this run as a
pipeline dry-run, not a real trajectory experiment. For substantive
experiments, seek clips with clear translational camera motion (scope pushed
forward through a lumen, or sweeping across a surface) — or move to public
datasets like C3VD / EndoMapper as a first-round baseline.

---

## 10. Next steps after this test

Once the pipeline is verified end-to-end:

1. Batch-evaluate all your clips with an optical-flow / pulsation report
   (a `scripts/check_video.py` tool can be added — ask when you're ready).
2. Fine-tune DROID-SLAM's feature/context net on endoscope data
   (C3VD, SimCol3D, or EndoMapper) to close the domain gap.
3. Replace the assumed-FOV calibration in `calib.txt` with the real vendor
   intrinsics when they arrive.
