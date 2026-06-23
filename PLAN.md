# 胸腔內視鏡 3D 重建 + 軌跡估計 — 計畫書

最後更新：2026-06-22
主要素材：`lung_inner_76s.mov`（2940×1912, 30 fps, 76 s, 2273 frames）
參考論文：Soberanis-Mukul et al., *Monocular Vision-Based Endoscopic Sinus Navigation: A SLAM Driven Approach With CT Integration*, Healthcare Technology Letters, 2025.

---

## 1. 目標

**第一階段（本計畫範圍）**
- 從單眼內視鏡影片恢復：
  - 相機在世界座標下的 **6-DoF 軌跡**
  - 場景的 **稀疏 3D 點雲**（landmarks）
- 不做稠密 mesh、不做 4D 形變建模、不做 CT 對齊。
- 成功門檻：能 register ≥ 80% 的有效 frame，軌跡視覺上連續、無明顯漂移。

**後續階段（範圍外，本計畫不執行）**
- COLMAP MVS 或 Deformable 3DGS 稠密重建
- 呼吸 gating / 4D 重建
- 與術前 CT 的配準（如論文 §2.2 的 PnP-based init）

---

## 2. 技術選型（已確認）

| 元件 | 選擇 | 備註 |
|---|---|---|
| SLAM backbone | **DROID-SLAM** | 學習型，對 textureless 場景強；CUDA-only |
| 器械分割 | **SAM 2 + video propagation** | 第一個器械出現的 frame 手動點 1–3 個 prompt，propagate 全片 |
| 相機內參 | **自動估計** | 先用啟發式 `fx=fy≈W*0.85, cx=W/2, cy=H/2`，必要時用 COLMAP 精修 |
| 前處理 | OpenCV + FFmpeg | 圓形 FOV mask、bad-frame filter |
| 視覺化 | Open3D + matplotlib | 軌跡 + 稀疏點雲 |

---

## 3. 環境配置（Mac ↔ Windows 分工）

### Mac（開發 + 輕量步驟）
- macOS 系統，CPU + Apple Silicon GPU (MPS)
- 跑：影片抽幀、FOV mask、bad-frame filter、SAM 2 prompt 互動式選點、結果視覺化
- 開發/除錯 pipeline 程式碼

### Windows + RTX 3080 (10 GB VRAM)
- 跑：**SAM 2 video propagation（高解析下必須）**、**DROID-SLAM**（CUDA-only）
- 需要建立的環境：
  - Python 3.10
  - PyTorch 2.x + CUDA 11.8 或 12.1
  - DROID-SLAM repo（從 GitHub clone + 編譯 lietorch / droid_backends）
  - SAM 2 (`pip install git+https://github.com/facebookresearch/sam2.git`)
  - 模型權重：SAM 2 (`sam2_hiera_large.pt`) + DROID-SLAM (`droid.pth`)

### 工作流
```
[Mac] 抽幀 → FOV mask → filter → SAM2 prompt 選點 (prompts.json)
        │
        ▼
   rsync / 隨身碟 / OneDrive 同步整個 out/ 資料夾
        │
        ▼
[Win+3080] SAM2 propagate → 合成 mask → DROID-SLAM → 輸出 poses + points
        │
        ▼
   結果拉回 Mac
        │
        ▼
[Mac] Open3D 視覺化、檢視 trajectory.json
```

---

## 4. Pipeline 階段詳述

每個階段對應 `run.py` 的一個 subcommand，可獨立執行、可從中斷處 resume。

| # | 階段 | subcommand | 平台 | 預期耗時 (76 s 影片) |
|---|---|---|---|---|
| 0 | 環境檢查 | `check` | 任一 | < 5 s |
| 1 | 抽幀（含降採樣） | `extract` | Mac | ~30 s |
| 2 | 圓形 FOV mask | `fovmask` | Mac | < 5 s |
| 3 | Bad-frame filter | `filter` | Mac | ~30 s |
| 4a | SAM 2 prompt 互動式選點 | `sam2-prompt` | Mac | 人工 ~2 min |
| 4b | SAM 2 video propagate | `sam2-propagate` | **Win+3080** | ~5–10 min |
| 5 | 合成最終 mask | `combine-masks` | 任一 | < 10 s |
| 6 | 相機內參估計 | `calib` | Mac | < 5 s（heuristic）/ ~5 min（COLMAP refine） |
| 7 | DROID-SLAM 跑軌跡 | `droid` | **Win+3080** | ~10–20 min |
| 8 | 視覺化軌跡 + 點雲 | `viz` | Mac | < 5 s |

### 4.1 Stage 1 — 抽幀
- ffmpeg 抽出每一幀為 PNG（保 alpha-less RGB）
- **降採樣到 1280 寬**（DROID-SLAM 在 2940 寬下 10 GB VRAM 會炸）
- 同步輸出 `metadata.json`：原始 fps、frame count、原解析度、scale factor

### 4.2 Stage 2 — 圓形 FOV mask
- 取第一幀的 grayscale > 12 為有效區，做 morphological closing
- 輸出單張 `fov_mask.png`（整個影片共用）

### 4.3 Stage 3 — Bad-frame filter
- 對每一幀算：FOV 內平均亮度、specular pixel 比例
- 篩除：mean brightness > 140 或 specular > 5%（依先前量化分析訂的閾值）
- 輸出 `frame_metrics.csv` 與 `good_frames.txt`

### 4.4 Stage 4a — SAM 2 prompt 互動式選點
- 啟動 matplotlib GUI，顯示器械首次明顯出現的 frame（預設第 32 s, 44 s, 67 s 各一張）
- 左鍵 = 前景 (+1)，右鍵 = 背景 (–1)，每張 1–3 個點
- 存成 `prompts.json`

### 4.5 Stage 4b — SAM 2 video propagation
- 載入 `sam2_hiera_large.pt`
- 從 prompt frame 向前 + 向後 propagate
- 輸出 `tool_masks/000000.png ... NNNN.png`（白 = 器械）

### 4.6 Stage 5 — 合成最終 mask
- 對每張 good frame 算 `final = FOV ∩ NOT(tool)`
- 輸出 `final_masks/`，這是 DROID-SLAM 的輸入 mask

### 4.7 Stage 6 — 相機內參估計
- 預設 heuristic：fx = fy = W × 0.85，cx = W/2，cy = H/2
- 可選：先用 COLMAP 跑前 200 幀做 auto-init，取它估出的 K
- 輸出 `calib.json`

### 4.8 Stage 7 — DROID-SLAM
- 透過子程序呼叫 DROID-SLAM repo 的 `demo.py`
- 傳入 `--calib calib.txt`、`--imagedir frames`、`--mask final_masks`（fork 版本支援；官方版需小幅修改）
- 輸出 `poses.npy`、`disps.npy`、`tstamps.npy`、`points.ply`

### 4.9 Stage 8 — 視覺化
- Open3D 讀取點雲 + poses，畫出 3D 軌跡（連線 + frame frusta）
- 另存 `trajectory.json`、`trajectory.png`、`viz_3d.html`（Plotly）

---

## 5. 預期問題與應對

| 預期問題 | 應對方案 |
|---|---|
| **DROID-SLAM 不接受 mask 輸入（官方版）** | 兩個選項：(a) 把 mask 區直接塗成 mean color，騙過特徵 (b) 改 fork 版（如 [DROID-SLAM-mask] 或自己 patch dvideo.py） |
| **呼吸形變導致 BA 不收斂** | 第一階段不處理；若整個影片不收斂，把影片切成 ~5 s 短段分別跑，最後拼接 |
| **多段子軌跡 / tracking loss** | 偵測 DROID-SLAM 輸出的 `tstamps.npy` 不連續處 → 自動切 segment |
| **K 估錯導致軌跡有「香蕉變形」** | 對比 heuristic vs COLMAP-refined K 兩組結果，選軌跡更平滑的 |
| **SAM 2 mask 在器械邊緣抖動** | 對 mask 做 temporal smoothing（5-frame morphological dilation 取 union） |
| **過曝 frame 拖累 SLAM** | Stage 3 已先 filter；如果還是出問題，調嚴閾值 |
| **VRAM 不足（10 GB）** | (a) 降到 960 寬 (b) 縮短 sliding window (c) 切 segment |

---

## 6. Milestones / 時程

| 週次 | 里程碑 | 交付物 |
|---|---|---|
| W1 | 環境配置完成（Mac 與 Win 雙端） | `python run.py check` 雙端都全綠 |
| W1 | Stage 1–3 跑通 | `out/frames/`, `fov_mask.png`, `good_frames.txt` |
| W2 | Stage 4 SAM 2 跑出器械 mask | `out/tool_masks/` |
| W2 | Stage 5–6 完成 | `final_masks/`, `calib.json` |
| W3 | DROID-SLAM 第一次跑通 | `poses.npy`, `points.ply` |
| W3 | 軌跡可視化 | `trajectory.png`, 3D Open3D viewer |
| W4 | 與光學追蹤 / 手動 GT 對比評估 | 軌跡誤差報告（若有 GT） |

---

## 7. 評估指標（無 ground truth 時的代用指標）

1. **Frame registration rate**：DROID-SLAM 成功 register 的 frame 數 / 全片有效 frame 數
2. **Trajectory smoothness**：相鄰 pose 之間的速度 / 角速度有沒有異常跳躍
3. **Loop closure consistency**：若鏡頭來回掃過同一區域，看點雲是否對齊
4. **Visual sanity check**：把 sparse 點雲投影回影像，檢查是否落在合理組織上

若日後取得 NDI Polaris 或機械臂 GT，可導入 translation/rotation error（如論文表 1 的 3.2 mm / 4.9°）。

---

## 8. 風險與備案

| 風險 | 機率 | 影響 | 備案 |
|---|---|---|---|
| DROID-SLAM 在這支影片完全不收斂 | 中 | 高 | 退回 COLMAP SfM；先 register 一個 sub-segment |
| 呼吸形變超出容忍 | 中 | 高 | 加 frame gating；或直接跳到 4D 方法（Deformable 3DGS） |
| 3080 VRAM 不足 | 低 | 中 | 降解析度到 960 寬 |
| SAM 2 mask 不夠乾淨 | 低 | 低 | 手動修少數幾張關鍵幀；或加 morphological dilation |
| 沒有相機原機可校正 | 中 | 中 | 用 COLMAP self-cal；接受 ~10% 焦距誤差 |

---

## 9. 開發者備忘

- 程式碼 entry：`python run.py <subcommand>`，全部 subcommand 列表執行 `python run.py --help`
- 每個 stage 的輸出都進 `out/` 子目錄，可獨立刪除重跑
- 進度訊息會 stream 到 stdout，格式 `[HH:MM:SS | stage] message`
- 中斷 (Ctrl-C) 後重跑，已完成的 frame 會自動跳過

