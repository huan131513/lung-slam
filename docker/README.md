# Docker setup — endoscopic pipeline

Two images, one per heavy stage:

| Image | Base | Stage it runs | Why a separate image |
|---|---|---|---|
| `endo-sam`   | `pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime` | Stage 4b `sam2-propagate` | SAM 2 needs PyTorch ≥ 2.3 |
| `endo-droid` | `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`  | Stage 7 `droid`            | DROID-SLAM CUDA extensions need PyTorch 1.10 + nvcc 11.3 |

Other stages (`extract`, `fovmask`, `filter`, `sam2-prompt`, `combine-masks`, `calib`, `viz`) run on the **host** (your Mac or Ubuntu), no Docker required.

---

## 1. Prerequisites (Ubuntu host — once per machine)

```bash
# (a) Docker Engine
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker

# Add yourself to the docker group so you don't need sudo
sudo usermod -aG docker $USER
newgrp docker        # or log out / log in

# (b) NVIDIA Container Toolkit (lets containers see the GPU)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# (c) Smoke test — should print the host nvidia-smi from inside a CUDA container
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## 2. Build the two images

```bash
cd ~/path_navigate           # wherever you sync'd the project to
./docker/build.sh            # builds both (~ 15–25 min total, mostly DROID-SLAM compile)
# or selectively:
# ./docker/build.sh sam
# ./docker/build.sh droid
```

After build:
```bash
docker images | grep endo-
# endo-sam     latest   ...   ~ 7 GB
# endo-droid   latest   ...   ~ 8 GB
```

---

## 3. Prepare model weights (one-off)

Only SAM 2's weight needs to be downloaded manually; DROID-SLAM's weight is baked into the image.

```bash
mkdir -p ~/models
cd ~/models
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The `run.sh` script mounts `~/models` (or whatever `MODELS_DIR` you export) read-only at `/models` inside the container.

---

## 4. Daily usage

### Mac (host) — prep stages
```bash
cd ~/path_navigate
python run.py extract --video /path/to/lung_inner_76s.mov --out ./out --width 1280
python run.py fovmask --out ./out
python run.py filter  --out ./out
python run.py sam2-prompt --out ./out      # matplotlib GUI; pick tool prompts
python run.py calib --out ./out
```

### Sync `out/` and `run.py + pipeline/` to the Ubuntu+3090 box
```bash
rsync -avz --exclude='out/frames' ~/path_navigate/ ubuntu3090:~/path_navigate/
rsync -avz ~/path_navigate/out/frames/ ubuntu3090:~/path_navigate/out/frames/
```

### Ubuntu (3090) — heavy stages in Docker
```bash
cd ~/path_navigate

# Stage 4b: SAM 2 video propagation
./docker/run.sh sam sam2-propagate \
    --out ./out \
    --ckpt /models/sam2.1_hiera_large.pt \
    --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml

# Stage 5: combine FOV + tool masks (light, but easiest to keep in SAM image)
./docker/run.sh sam combine-masks --out ./out

# Stage 7: DROID-SLAM
./docker/run.sh droid droid --out ./out --stride 1
```

### Sync results back to Mac (for viz)
```bash
rsync -avz ubuntu3090:~/path_navigate/out/ ~/path_navigate/out/
```

### Mac — visualize
```bash
python run.py viz --out ./out
```

---

## 5. Volume layout inside the container

```
/workspace/                  ← host: ~/path_navigate/   (rw, owned by you)
  ├── run.py
  ├── pipeline/
  ├── out/                   ← results land here on host
  └── ...

/models/                     ← host: ~/models/  (read-only)
  └── sam2.1_hiera_large.pt

/opt/DROID-SLAM/             ← inside endo-droid image only (not on host)
  ├── droid.pth              ← weights baked in
  └── demo.py                ← invoked by pipeline/droid_runner.py
```

The pipeline already knows to look at `DROID_SLAM_ROOT=/opt/DROID-SLAM` because the Dockerfile exports it.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit not installed / docker not restarted. Re-run step 1(b). |
| `Permission denied` on `out/` files | Re-run with `--user "$(id -u):$(id -g)"` — already in `run.sh`. If still bad, check that the host dir is writable. |
| SAM 2 OOM on 3090 | Drop frame width in stage 1: `python run.py extract --width 960 ...` and redo. |
| DROID-SLAM tracking loss mid-run | Add `--stride 2` to `./docker/run.sh droid droid ...`. |
| Want to poke around interactively | `./docker/run.sh sam shell` or `./docker/run.sh droid shell`. |
| Image build cache stale | `./docker/build.sh sam` only rebuilds one; or `docker build --no-cache -f Dockerfile.sam -t endo-sam docker/` |

---

## 7. Cleaning up

```bash
# Remove images (frees ~ 15 GB)
docker rmi endo-sam endo-droid

# Remove dangling intermediate layers
docker system prune

# Remove the project (host)
rm -rf ~/path_navigate
```

Nothing else touches the host system.
