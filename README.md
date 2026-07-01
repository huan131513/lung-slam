# lung-slam

Endoscopic 3D reconstruction pipeline — recover camera trajectory + sparse 3D
points from monocular thoracoscopic video.

Plan: [`PLAN.md`](PLAN.md) · Docker setup: [`docker/README.md`](docker/README.md)

---

## Pipeline at a glance

Two paths are supported. **The direct DROID-SLAM path is recommended for
endoscope video** — DROID-SLAM does its own optical-flow-based keyframe
selection internally, so pre-filtering by brightness / specular tends to
throw away too many frames.

### A) Direct DROID-SLAM path (recommended)

```
preprocess  →  droid  →  viz
```

`preprocess` does: ffmpeg extract → circular-FOV detection → shrink radius
to drop peripheral distortion → crop to inscribed square → emit DROID-SLAM
`calib.txt`. No brightness / specular filter, no SAM 2, no `--mask_dir`.
100% compatible with the official
[DROID-SLAM `demo.py`](https://github.com/princeton-vl/DROID-SLAM).

```bash
python run.py all-droid --video data/endo.mp4 --out ./out
# or step-by-step:
python run.py preprocess --video data/endo.mp4 --out ./out
python run.py droid      --out ./out                       # --no-masks by default
python run.py viz        --out ./out
```

### B) Full pipeline with SAM 2 tool masking (advanced)

```
Stage 1 extract           ─┐
Stage 2 fovmask            │  CPU-only
Stage 3 filter             │  ⚠ tuned for other footage; often too aggressive
Stage 4a sam2-prompt       │  Interactive matplotlib
Stage 6 calib             ─┘

Stage 4b sam2-propagate   ─┐  CUDA required → endo-sam   docker image
Stage 5 combine-masks      │
Stage 7 droid --use-masks ─┘  requires a DROID-SLAM fork that accepts --mask_dir

Stage 8 viz                  Open3D / matplotlib
```

Every stage prints timestamped progress; each can be run independently:
```bash
python run.py <stage> --out ./out [stage args]
python run.py --help        # list all stages
```

---

## Cross-machine workflow

```
[Mac]                          [GitHub]              [Ubuntu + RTX 3090]
─────                          ────────              ────────────────────
edit code  ──── git push ────►  ◄──── git pull ────  build docker images
                                                     (see docker/README.md)
extract / fovmask / filter
sam2-prompt (GUI)
calib
                                                     sam2-propagate
        ──────── rsync out/ and data/ ─────────►     combine-masks
                                                     droid
        ◄─────── rsync out/ back ───────────────
viz
```

Video files and model weights are **not** committed — transfer via `rsync` or
`scp`. See `.gitignore`.

---

## Quick start on Ubuntu (after `git clone`)

### Direct DROID-SLAM path (Path A)

```bash
# 1. Install DROID-SLAM from upstream (once per machine)
git clone --recursive https://github.com/princeton-vl/DROID-SLAM ~/DROID-SLAM
cd ~/DROID-SLAM
conda env create -f environment.yaml
conda activate droidenv
python setup.py install
./tools/download_model.sh           # -> ~/DROID-SLAM/droid.pth
export DROID_SLAM_ROOT=~/DROID-SLAM

# 2. Install this repo's Python deps (ffmpeg, opencv, open3d, tqdm ...)
cd ~/path_navigate
pip install -r requirements.txt

# 3. Put the video somewhere and run the direct path
mkdir -p data && cp /path/to/endo.mp4 data/
python run.py all-droid --video data/endo.mp4 --out ./out \
    --fps 15 --distortion-margin 0.15 --assumed-fov-deg 90
```

Outputs land in `out/`:
```
out/frames/          cropped rectangular PNGs fed to DROID-SLAM
out/calib.txt        "fx fy cx cy" (rough default; replace when vendor calib arrives)
out/preview.png      overlay: yellow=raw FOV, red=distortion-trimmed, green=final crop
out/droid/           DROID-SLAM reconstruction (poses.npy, disps.npy, ...)
out/viz/             trajectory + point cloud renders
```

### Path B (full pipeline with Docker + SAM 2)

```bash
./docker/build.sh
mkdir -p ~/models
MODELS_DIR=~/models bash scripts/download_weights.sh
./docker/run.sh sam   sam2-propagate --out ./out --ckpt /models/sam2.1_hiera_large.pt
./docker/run.sh sam   combine-masks  --out ./out
./docker/run.sh droid droid          --out ./out --use-masks    # needs a fork
```

Full instructions, prerequisites and troubleshooting are in
[`docker/README.md`](docker/README.md).

---

## File layout

```
.
├── PLAN.md                  full project plan
├── README.md                this file
├── requirements.txt         pure-CPU host deps
├── run.py                   CLI entry point
├── pipeline/                stage modules (see PLAN.md §4)
├── docker/                  CUDA images for the heavy stages
│   ├── Dockerfile.sam       SAM 2 (PyTorch ≥ 2.3, CUDA 12.1)
│   ├── Dockerfile.droid     DROID-SLAM (PyTorch 1.10, CUDA 11.3)
│   ├── build.sh             build both
│   ├── run.sh               run a stage in either container
│   └── README.md            Linux setup + daily usage
└── scripts/
    └── download_weights.sh  SAM 2 weight only
```

Outputs land in `out/` (gitignored).
