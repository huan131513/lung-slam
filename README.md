# lung-slam
0. 每次開新 terminal 都要先做
   
```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate droidenv
export DROID_SLAM_ROOT=$HOME/DROID-SLAM
cd ~/lung-sla
```

1. 放檔案 + 決定資料夾命名

把影片放到 data/，--out 用一個跟影片對應的獨立名稱（每支影片一個資料夾，不要共用）：

`cp /path/to/your_video.mov ~/lung-slam/data/`

2. preprocess（擷取影格 + FOV 裁切 + 自動產生 calib.txt）

```
python run.py preprocess \
  --video data/your_video.mov \
  --out ./out/your_video \
  --fps 15
```

會產生 out/your_video/{frames/, calib.txt, preview.png, metadata.json}。

常用可調參數：
- --fps：擷取後的影格率，預設 15（設 0 = 保留原始 fps，通常太密不必要）
- --width：縮圖寬度，預設 1280
- --distortion-margin：內視鏡圓形視野邊緣再往內縮的比例，預設 0.15（邊緣扭曲嚴重可加大）
- --assumed-fov-deg：假設的水平視角，預設 90，這個直接影響 calib.txt 的準確度

檢查 preview.png：確認裁切後的圓形視野有正確蓋住有效畫面、沒有黑邊殘留，這步沒做好會直接拖累 DROID-SLAM 的追蹤品質。

關於 calib.txt 的準確度：這是用「假設的視角角度」+ 偵測到的圓形半徑換算出來的粗略估計值（metadata.json 裡也會註記 "calib.txt is a rough estimate"）。如果你手上有內視鏡廠商提供的真實內參（fx fy cx cy），直接覆寫 out/your_video/calib.txt（格式是一行 fx fy cx cy）取代這個估計值，重建的尺度/準確度會更好。

2.1 想要擷取子片段

只跑整支影片裡的某一段幀數範圍（例如第 1490～1940 幀），用 split_video.py，不是 run.py preprocess（後者沒有 --start/--end，只能整支處理）：

```
python split_video.py \
  --video data/your_video.mp4 \
  --start 1490 \
  --end 1940 \
  --out ./out/your_video_1490-1940
```

- --start / --end：幀數索引（0-based，含頭含尾）
- 其他參數（--width、--fps、--distortion-margin、--assumed-fov-deg、--intrinsics、--calib-model、--no-real-calib）跟 preprocess 相同，但 --fps 預設值不同：split_video.py 預設 0（保留原始 fps），preprocess 預設降到 15，想比照平常流程要自己加 --fps 15
- 輸出格式跟 preprocess 一樣（frames/、calib.txt、preview.png、metadata.json），接著照常跑 3./4. 步驟即可：
  python run.py droid --out ./out/your_video_1490-1940
  python run.py viz-live --out ./out/your_video_1490-1940
  
3. droid（跑 DROID-SLAM）

`python run.py droid --out ./out/your_video`

跑的過程中會自動跳出「Droid Visualizer」視窗（官方即時預覽，跑完自動關），不用靠它判斷品質好壞，等終端機印出 outputs: ... 才算真正跑完。

如果影片動作幅度小、keyframe 太少（可以先跑一次看 viz-live log 印出的 keyframe 數），可以調低這兩個門檻拿到更多 keyframe：
python run.py droid --out ./out/your_video --filter-thresh 1.5 --keyframe-thresh 3.0

4. viz-live（檢查最終結果）

`python run.py viz-live --out ./out/your_video`
- 拖曳滑鼠旋轉、滾輪縮放
- P 暫停/繼續、R 重播、S/A 放寬/收緊濾波
- 如果畫面看起來像「多片 2D 貼片脫節」，通常代表深度尺度不穩定，先試試 --filter-count 4：
`python run.py viz-live --out ./out/your_video --filter-count 4`

一行懶人版（等同 preprocess+droid+viz 三步，仍建議跑完後另外執行 viz-live）

`python run.py all-droid --video data/your_video.mov --out ./out/your_video`



Endoscopic 3D reconstruction pipeline — recover camera trajectory + sparse 3D
points from monocular thoracoscopic video.

Plan: [`PLAN.md`](PLAN.md) · **Ubuntu quickstart: [`UBUNTU_RUN.md`](UBUNTU_RUN.md)** · Docker setup: [`docker/README.md`](docker/README.md)

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
