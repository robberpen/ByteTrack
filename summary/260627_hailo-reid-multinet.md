# Hailo 多網路 HEF:YOLOX-S + ReID(OSNet)合成一顆 HEF

> 紀錄日期:2026-06-27
> 主軸:`summary/hailo8_rpi5_deploy.md`、前篇 `summary/260627_hailo-nms-endnodes.md`
> 本篇:把 ReID 模型轉 ONNX、備校準 crop、用 `ClientRunner.join` 編成單一多網路 HEF

---

## 故事 / 背景(What & Why)

ReID-based MOT(DeepSORT/StrongSORT/BoT-SORT)是雙模型:YOLOX 偵測 + ReID 抽特徵。
若兩者是**兩個獨立 HEF**,每幀在晶片上來回 `configure()`(從 host 重載)→ context-switch
延遲大。傳言「融合成單模型雙輸入雙輸出」其實是**半真**:det+ReID 因 ReID 輸入是偵測輸出
裁出的 crop(中間夾 CPU 步驟),**不可能**併成一張前向圖。正確做法是用 Hailo 的
**多網路 HEF**:兩個獨立網路編進**同一顆 HEF**,device 只 configure 一次、共同常駐,
每幀只在 group 間 `activate`(便宜)。仍是兩次推理,但省掉重載延遲。

ReID 半邊選 **OSNet**(`osnet_ain_ms_d_c.pth.tar`,8.7MB ONNX),不選 SBS_S50
(ResNet50,336MB)——後者塞進單顆 hailo8l 的資源風險高。

相關環境:DFC `hailo_sdk_client` 3.29.0、torch 2.4.1+cu121、RTX 3070(8GB)。
跨兩個 repo:
- ByteTrack:`/sera/share2507/AI/ultralytics_poetry/ByteTrack`
- FDOS(ReID 來源):`/sera/AOSP-G520/MOT/ultralytics_poetry_fdos/Fast-Deep-OC-SORT`

---

## 思路重點

- **`ClientRunner.join(runner2, scope1_name, scope2_name, join_action=JoinAction.NONE)`**
  是官方多網路 API(client_runner.py:1280)。獨立兩網路用 `JoinAction.NONE`(不接線)。
- **狀態規則**:兩 runner 必須同狀態。獨立網路 → **各自先量化好(各自校準集)再 join**
  (join 後 compile)。scope 名不可重疊(`yolox` / `osnet_reid`)。
- **OSNet 正規化**:torchreid 用 ImageNet norm 訓練 → 烤 `normalization(mean*255, std*255)`,
  與 YOLOX 同常數。`embedding.py` 那段餵 raw 0-255 看似 latent bug,部署改用正確 norm。
- **校準 crop** 餵原始 0-255 RGB(不 resize/normalize),resize 384x128 與 on-chip norm
  都交給 compile 端 `build_crop_calib`。

---

## Workflow — step by step

### Step 1 — ReID 權重轉 ONNX(在 FDOS repo)
腳本:`export_osnet_onnx.py`(repo 根目錄,新增)
```bash
cd /sera/AOSP-G520/MOT/ultralytics_poetry_fdos/Fast-Deep-OC-SORT
python3 export_osnet_onnx.py
```
- importlib 直接載 `external/deep-person-reid/torchreid/models/osnet_ain.py`(避開需 torchvision 的整包)
- `osnet_ain_x1_0(num_classes=2510, loss="softmax", pretrained=False)`
- 載 `external/weights/osnet_ain_ms_d_c.pth.tar` 的 state_dict,去 `module.` 前綴 → load(missing=0/unexpected=0)
- `eval()`(回傳 512-d embedding,跳過分類層)→ `torch.onnx.export`
- 產物:`osnet_ain_x1_0.onnx`(8.7MB),input `images[batch,3,256,128]`、output `embeddings[batch,512]`、opset 12、dynamic batch
  (原本 384x128 會撐爆量化,改回官方標準 256x128 — 見下方疑難排解)

### Step 2 — 備 ReID 校準 crop(在 ByteTrack repo)
腳本:`tools/crop_mot20_reid.py`(新增)
```bash
cd /sera/share2507/AI/ultralytics_poetry/ByteTrack
python3 tools/crop_mot20_reid.py --mot-root datasets/MOT20/train --out datasets/reid_crops
```
- 逐序列 `np.loadtxt(<seq>/gt/gt.txt, delimiter=',')`
- 過濾(沿用 `convert_mot20_to_coco.py:97-127`):`ignore(ann[6])==1`、class 行人
  (排除非人 `[3,4,5,6,9,10,11]`、ignored `[2,7,8,12]`)、`visibility>=0.3`、`h>=64`、每 20 幀取一次
- 讀 `img1/<frame:06d>.jpg`,bbox clip 邊界後裁切;同影像只解碼一次
- 存 `datasets/reid_crops/<seq>_f<frame>_id<track>.jpg`,上限 4000(超過隨機抽樣)
- 實跑:候選約 40000(01:596 / 02:4240 / 03:12140 / 05:23715)→ 寫出 **4000 張**

### Step 3 — 多網路編譯成單一 HEF(在 ByteTrack repo)
腳本:`tools/hailo_compile.py`(新增 `--second-*` / `--first-scope` 選項)
```bash
python3 tools/hailo_compile.py \
    --onnx bytetrack_s_mot20.onnx --calib-dir datasets/MOT20/train \
    --hw-arch hailo8l --output bytetrack_yolox_osnet.hef --finetune-batch 4 \
    --second-onnx /sera/AOSP-G520/MOT/ultralytics_poetry_fdos/Fast-Deep-OC-SORT/osnet_ain_x1_0.onnx \
    --second-calib-dir datasets/reid_crops
```
內部:
1. YOLOX parse → model script(norm + 可選 nms/finetune)→ `optimize` 量化(MOT 整幀校準)
2. `build_reid_runner()`:ReID parse(end-node 自動偵測)→ 烤 ImageNet norm → crop 校準 → `optimize` 量化
3. `runner.join(r2, scope1_name="yolox", scope2_name="osnet_reid", join_action=JoinAction.NONE)`
4. `compile()` → 單一多網路 HEF(另存 `_joined_quant.har`)

### Step 4 — Pi 部署(尚未實作)
`deploy/Hailo/hailo_inference.py` 目前只跑單網路;多網路 HEF 需用 HailoRT 取兩個 network group
分別推理(YOLOX 整幀 → crop → ReID),全程不重配 device。

---

## Checkpoint(已驗證)

- `osnet_ain_x1_0.onnx`:onnx.checker OK、ORT batch=2 → `(2,512)`、8.7MB。
- `datasets/reid_crops`:4000 張,crop 為正常行人比例(如 201x80、245x104)。
- 消費端 `build_crop_calib('datasets/reid_crops',384,128,...)` → `(N,384,128,3)` float32、值域 0–255。

---

## 關鍵檔案

| 檔案 | 角色 |
|------|------|
| `Fast-Deep-OC-SORT/export_osnet_onnx.py` | Step 1:OSNet→ONNX(新增) |
| `Fast-Deep-OC-SORT/trackers/ocsort_embedding/embedding.py:87-113` | OSNet 載入/前處理 recipe 來源 |
| `tools/crop_mot20_reid.py` | Step 2:MOT20 GT→crop(新增) |
| `tools/convert_mot20_to_coco.py:97-127` | GT 過濾慣例來源 |
| `tools/hailo_compile.py` | Step 3:多網路 join 編譯(`build_reid_runner`/`build_crop_calib`/`--second-*`) |
| `hailo_sdk_client/runner/client_runner.py:1280` | `ClientRunner.join` API |

---

## ReID 量化疑難排解(已實測解決)

OSNet ReID 量化時連環兩個錯,皆已解:

### 1. global avgpool int8 溢位
```
AccelerasNumerizationError: Shift delta in osnet_reid/avgpoolN/avgpool_op is
larger than 2, cannot quantize. ... reduce global average-pool spatial dimensions
```
- 根因:OSNet-AIN 有大量 global-avgpool aggregation gate;384x128 大輸入更糟。
- 解 A:**改用官方標準 256x128 重匯出**(Model Zoo `osnet_x1_0` 此尺寸量化乾淨;
  `export_osnet_onnx.py` 已改 `INPUT_H,INPUT_W=256,128`)。Pi 端 crop 要 resize 256x128。
- 解 B(仍需):對**每個 global avgpool**(output 1x1)套
  `pre_quantization_optimization(global_avgpool_reduction, layers=avgpoolN, division_factors=[H,1])`
  (H=該層輸入高;`[H,1]` 即 DFC 自動模式的拆法)。**只挑 output 1x1**,
  不動 transition 的 downsample avgpool。`build_reid_runner` 會自動列舉 HN 套用(實測 35 顆)。
- 正確語法重點:`layers=avgpoolN`(bare、無 scope、不能 `*` glob)、需 `division_factors`。

### 2. bias_correction zero-size array
- 預設 optimization level 跑到 bias_correction 丟
  `ValueError: zero-size array to reduction operation maximum`。
- 解:ReID 用 **optimization_level=0**(已實測 `PASS_LEVEL0`)。
  `--second-optimization-level` 預設 0,與 YOLOX 的 level/finetune 解耦。
- 代價:level 0 跳過 bias_correction/finetune,精度略降;higher level 修復列為 TODO。

> 驗證:256x128 + 35 個 avgpool reduction + level 0 → ReID 完整量化 OK(小校準 128 張)。

## Risk / TODO

- [ ] **資源是否塞得下 hailo8l**:YOLOX-S@608x1088 + OSNet 兩網路同顆 HEF,compile 可能失敗或 FPS 掉;
      只能實際編才知道(OSNet 小,風險比 SBS_S50 低很多)。
- [ ] **實跑 Step 3** 多網路編譯(兩次量化 + join,吃 GPU 時間);OOM 再降 `--finetune-batch 2`。
- [ ] **OSNet 正規化驗證**:烤 ImageNet norm vs `embedding.py` 餵 raw 0-255 的差異,
      上 Pi 用原 PyTorch 特徵對照。
- [ ] **Step 4**:`hailo_inference.py` 擴充成讀多網路 HEF、跑兩個 network group。
- [ ] (選用)若要改用 SBS_S50:需先用 FastReID export ONNX,且可能要改 hailo8 才塞得下。
- [ ] crop 多樣性 > 數量;`--frame-stride`/`--min-vis` 可依分佈再調。
