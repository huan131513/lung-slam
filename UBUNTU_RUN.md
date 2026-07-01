# Ubuntu Quickstart — Direct DROID-SLAM Path

Step-by-step instructions to reproduce a single-clip trajectory reconstruction
on Ubuntu + RTX 3090, using the direct path (no filter, no SAM 2, no `--mask_dir`).

Verified on: Ubuntu 22.04, CUDA 11.8+, PyTorch 1.10+.

---

## 0. Assumptions

- You already have the repo cloned at `~/path_navigate` (or clone it now).
- You have Conda / Miniconda installed.
- You have an NVIDIA driver + CUDA runtime installed on the host.
- Your video file is somewhere on the host filesystem.

---

## 1. Pull latest code

```bash
cd ~/path_navigate
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

```bash
cd ~
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM
conda env create -f environment.yaml
conda activate droidenv
python setup.py install
./tools/download_model.sh          # downloads droid.pth (~250 MB) into repo root
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

Inside the `droidenv` conda env (or a separate env — either works):

```bash
cd ~/path_navigate
pip install -r requirements.txt
sudo apt-get install -y ffmpeg      # if not already present
```

Sanity check:
```bash
python run.py check
```
Should print Python / CUDA / ffmpeg / OpenCV versions with no errors.

---

## 4. Prepare the test clip

The 10-second sample clip is committed to the repo at
[`data/7348331-618-628.mp4`](data/7348331-618-628.mp4) (~10 MB, 1920×1080 @ 60 fps).
After `git pull` in Step 1, it is already on disk — nothing else to download.

Verify:
```bash
cd ~/path_navigate
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
├── preview.png       # crop overlay (yellow=raw FOV, red=trimmed, green=final crop)
├── calib.txt         # "fx fy cx cy"
├── frames/           # ~20 cropped PNGs (5s × 4fps)
├── metadata.json
├── droid/            # DROID-SLAM output
│   ├── poses.npy     # (N, 7)  camera poses [tx ty tz qx qy qz qw]
│   ├── disps.npy     # per-keyframe inverse depth
│   ├── tstamps.npy   # keyframe indices
│   └── images.npy    # keyframe images
└── viz/              # trajectory + point cloud visualizations
```

Sanity checks:

| Check | Command | Pass criterion |
|---|---|---|
| Preview crop is correct | open `preview.png` | green box is inside the tissue view, no black border |
| Number of frames | `ls out/clip_5to10/frames \| wc -l` | ~20 |
| DROID converged | `python -c "import numpy as np; p=np.load('out/clip_5to10/droid/poses.npy'); print(p.shape)"` | shape starts with a number > 10 |
| Trajectory shape | open `out/clip_5to10/viz/trajectory.png` (or use Open3D) | should look like a short arc, not a jittery zigzag |

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
