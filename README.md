# lung-slam

Endoscopic 3D reconstruction pipeline — recover camera trajectory + sparse 3D
points from monocular thoracoscopic video.

Plan: [`PLAN.md`](PLAN.md) · Docker setup: [`docker/README.md`](docker/README.md)

---

## Pipeline at a glance

```
Stage 1 extract           ─┐
Stage 2 fovmask            │  CPU-only, runs on Mac or Ubuntu host
Stage 3 filter             │
Stage 4a sam2-prompt       │  Interactive matplotlib (do on Mac)
Stage 6 calib             ─┘

Stage 4b sam2-propagate   ─┐  CUDA required → endo-sam   docker image
Stage 5 combine-masks      │
Stage 7 droid             ─┘  CUDA required → endo-droid docker image

Stage 8 viz                  back on Mac (Open3D / matplotlib)
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

```bash
# 1. Build the two Docker images (~15–25 min the first time)
./docker/build.sh

# 2. Download SAM 2 weight (DROID-SLAM weight is already baked into the image)
mkdir -p ~/models
MODELS_DIR=~/models bash scripts/download_weights.sh

# 3. Drop the video into ./out's sibling and prep on the host (or sync from Mac)
mkdir -p data
# scp ~/Downloads/lung_inner_76s.mov ubuntu3090:~/path_navigate/data/

# 4. Heavy stages (Docker)
./docker/run.sh sam   sam2-propagate --out ./out --ckpt /models/sam2.1_hiera_large.pt
./docker/run.sh sam   combine-masks  --out ./out
./docker/run.sh droid droid          --out ./out
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
