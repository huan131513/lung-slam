# lung-slam

Endoscopic 3D reconstruction pipeline — recover camera trajectory + sparse 3D
points from monocular thoracoscopic video.

Plan: [`PLAN.md`](PLAN.md) · **Ubuntu quickstart: [`UBUNTU_RUN.md`](UBUNTU_RUN.md)** · Docker setup: [`docker/README.md`](docker/README.md)

---

## 快速開始

### 0. 每次開新 terminal 都要先做

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate droidenv
export DROID_SLAM_ROOT=$HOME/DROID-SLAM
cd ~/lung-slam
```

### 1. 放影片

`--out` 用一個跟影片對應的獨立名稱（每支影片一個資料夾，不要共用）：

```bash
cp /path/to/your_video.mov ~/lung-slam/data/
```

### 2. 照情境選一個指令跑

| 情境 | 指令 |
|---|---|
| 整支影片，正常跑 | `python run.py all-droid --video data/xxx.mp4 --out ./out/xxx` |
| 整支影片，先濾掉過曝/高光的爛幀 | `python run.py all-clean --video data/xxx.mp4 --out ./out/xxx` |
| 只要某段幀數範圍（例如第 1490～1940 幀），濾爛幀 | `python run.py all-clean --video data/xxx.mp4 --start 1490 --end 1940 --out ./out/xxx_1490-1940` |
| 只要某段幀數範圍，不濾爛幀（最快） | `python split_video.py --video data/xxx.mp4 --start 1490 --end 1940 --out ./out/xxx_1490-1940` |

不確定要不要濾爛幀就先跑 `all-droid`（DROID-SLAM 自己會做動作篩選，通常不需要另外濾）；如果重建結果一堆雜訊/飄浮點，再試 `all-clean`。`all-clean` 的過濾門檻是抓其他影片調的，這支內視鏡片段常常濾掉六成以上，跑完可以看 `out/xxx/frame_metrics.csv` 決定要不要用 `--max-brightness`/`--min-brightness`/`--max-specular` 調鬆。

兩個 `all-*` 指令也都接受 `droid` stage 的門檻參數（keyframe 太少時用，見下面「篩選 keyframe」說明）：

```bash
python run.py all-clean --video data/xxx.mp4 --out ./out/xxx --filter-thresh 1.5 --keyframe-thresh 3.0
```

### 3. 看結果

```bash
python run.py viz-live --out ./out/xxx
```
- 拖曳滑鼠旋轉、滾輪縮放
- `P` 暫停/繼續、`R` 重播、`S`/`A` 放寬/收緊濾波
- 如果畫面看起來像「多片 2D 貼片脫節」，通常代表深度尺度不穩定，先試試 `--filter-count 4`

---

## 手動逐步執行（除錯 / 需要中途調參數時用）

上面的一行指令都只是把下面幾步串起來；想單獨重跑某一步、或某步失敗要重試，用這裡。

### preprocess（擷取影格 + FOV 裁切 + 自動產生 calib.txt）

```bash
python run.py preprocess \
  --video data/your_video.mov \
  --out ./out/your_video \
  --fps 15
```

會產生 `out/your_video/{frames/, calib.txt, preview.png, metadata.json}`。

常用可調參數：
- `--fps`：擷取後的影格率，預設 15（設 0 = 保留原始 fps，通常太密不必要）
- `--width`：縮圖寬度，預設 1280
- `--distortion-margin`：內視鏡圓形視野邊緣再往內縮的比例，預設 0.15（邊緣扭曲嚴重可加大）
- `--assumed-fov-deg`：假設的水平視角，預設 90，只在沒有真實內參可用時影響 calib.txt 準確度

檢查 `preview.png`：確認裁切後的圓形視野有正確蓋住有效畫面、沒有黑邊殘留，這步沒做好會直接拖累 DROID-SLAM 的追蹤品質。

**關於 calib.txt 的準確度**：目前預設會自動套用真實 checkerboard 內參（`data/intrinsics.json`，僅當來源影片原生解析度為 1920×1080 時適用）；解析度不符時才會退回用「假設視角角度＋偵測到的圓形半徑」換算出的粗略估計值（`metadata.json` 會標註 `calib_source` 是 `real:...` 還是 heuristic）。想強制用舊的估計值可加 `--no-real-calib`，想指定別的內參檔案用 `--intrinsics`。

#### 只要某段幀數範圍

`preprocess` 沒有 `--start`/`--end`，只能整支處理；要裁片段用 `split_video.py`：

```bash
python split_video.py \
  --video data/your_video.mp4 \
  --start 1490 \
  --end 1940 \
  --out ./out/your_video_1490-1940
```

- `--start` / `--end`：幀數索引（0-based，含頭含尾）
- 其他參數（`--width`、`--fps`、`--distortion-margin`、`--assumed-fov-deg`、`--intrinsics`、`--calib-model`、`--no-real-calib`）跟 `preprocess` 相同，但 `--fps` 預設值不同：`split_video.py` 預設 0（保留原始 fps），`preprocess` 預設降到 15，想比照平常流程要自己加 `--fps 15`
- 輸出格式跟 `preprocess` 一樣，接著照常跑下面的 droid/viz-live 步驟即可

### fovmask + filter（濾掉過曝/高光爛幀，`all-clean` 內部做的事）

```bash
python run.py fovmask --out ./out/your_video
python run.py filter  --out ./out/your_video
python run.py materialize-good --out ./out/your_video   # 把留下的乾淨幀複製到 frames_good/
```

`filter` 只看亮度 + 高光比例（不看模糊度），結果會印在終端機、也存進 `frame_metrics.csv`（每幀數值）跟 `good_frames.txt`（留下的幀 index）。門檻可調：`--max-brightness`（過曝，預設 140）、`--min-brightness`（太暗，預設 50）、`--max-specular`（高光反射比例，預設 0.05 = 5%）。

### droid（跑 DROID-SLAM）

```bash
python run.py droid --out ./out/your_video                  # 用 out/frames
python run.py droid --out ./out/your_video --use-filtered   # 改用 out/frames_good（要先跑完 filter + materialize-good）
```

跑的過程中會自動跳出「Droid Visualizer」視窗（官方即時預覽，跑完自動關），不用靠它判斷品質好壞，等終端機印出 `outputs: ...` 才算真正跑完。

**篩選 keyframe**：`--filter-thresh`（預設 2.4，決定一幀有沒有資格被考慮）跟 `--keyframe-thresh`（預設 4.0，決定收進來的幀要不要因為跟鄰居太像而被刪）都是純粹看 DROID 網路估出來的光流大小，跟畫質（模糊/高光）無關。如果影片動作幅度小、keyframe 太少，可以調低這兩個門檻拿到更多 keyframe：

```bash
python run.py droid --out ./out/your_video --filter-thresh 1.5 --keyframe-thresh 3.0
```

注意：如果 keyframe 之間仍有長時間的大空洞，通常不是門檻問題，而是那段時間鏡頭本身動得太少（軟組織蠕動為主、沒有真實平移）——這種情況再降門檻只會生出退化的重複幀，對重建沒有幫助，該考慮換一段動作幅度更大的片段。想「只保留乾淨幀、其餘全當 keyframe」在物理上也行不通：相機沒移動的時段本來就沒有新的 3D 資訊可以三角測量，跟門檻無關。

### viz-live（檢查最終結果）

```bash
python run.py viz-live --out ./out/your_video --filter-count 4
```
`--filter-count`：至少幾個視角同意才保留一個 3D 點，預設 2，畫面「多片脫節」時可試試調高。

---

## 指令總覽（一行懶人版）

| 指令 | 等同於 | 用途 |
|---|---|---|
| `run.py all-droid --video ... --out ...` | preprocess → droid → viz | 整支影片，直接跑（推薦起手式） |
| `run.py all-clean --video ... --out ...` | preprocess → fovmask → filter → materialize-good → droid → viz | 整支影片，先濾掉爛幀再跑 |
| `run.py all-clean --video ... --start N --end M --out ...` | 同上，限定幀數範圍 | 只測某一段 |
| `split_video.py --video ... --start N --end M --out ...` | 同 preprocess，限定幀數範圍 | 只裁片段，不跑 droid |

三個 `all-*` 指令跑完仍建議另外執行一次 `viz-live` 確認結果。

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

See the quick start above for the exact commands (`all-droid` / `all-clean`).

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

## First-time setup on a new Ubuntu machine

Once per machine — after this, use the quick start above. Full detail
(troubleshooting, version pins) is in [`UBUNTU_RUN.md`](UBUNTU_RUN.md).

```bash
# 1. Install DROID-SLAM from upstream
git clone --recursive https://github.com/princeton-vl/DROID-SLAM ~/DROID-SLAM
cd ~/DROID-SLAM
conda env create -f environment.yaml
conda activate droidenv
python setup.py install
./tools/download_model.sh           # -> ~/DROID-SLAM/droid.pth
export DROID_SLAM_ROOT=~/DROID-SLAM

# 2. Install this repo's Python deps (ffmpeg, opencv, open3d, tqdm ...)
cd ~/lung-slam
pip install -r requirements.txt

# 3. Sanity check
python run.py check
```

Outputs of the pipeline land in `out/<name>/`:
```
frames/          cropped rectangular PNGs fed to DROID-SLAM
frames_good/     (all-clean only) frames/ minus the ones filter dropped
calib.txt        "fx fy cx cy" (+ distortion if using real calibration)
preview.png      overlay: yellow=raw FOV, red=distortion-trimmed, green=final crop
frame_metrics.csv / good_frames.txt   (all-clean only) per-frame filter metrics + kept indices
droid/           DROID-SLAM reconstruction (poses.npy, disps.npy, ...)
viz/             trajectory + point cloud renders
```

### Path B (full pipeline with Docker + SAM 2)

Requires the Docker images from `docker/README.md` — see that file for
prerequisites and the full command sequence.

---

## File layout

```
.
├── PLAN.md                  full project plan
├── README.md                this file
├── requirements.txt         pure-CPU host deps
├── run.py                   CLI entry point
├── split_video.py           extract a [start,end] frame-range subclip
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
