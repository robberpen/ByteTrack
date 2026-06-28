# ByteTrack/YOLOX -> Hailo-8 @ RPi5 部署工作階段紀錄

## Context Header(供日後 reload)

- **Session**: `session_01YHDvBaxBy5Vp7Aavf7UfiK`
- **Date**: 2026-06-28
- **Repo**: `/media/peter/share2507/AI/ultralytics_poetry/ByteTrack` (branch: `devel`)
- **相關紀錄**:
  - [hailo8_rpi5_deploy.md](./hailo8_rpi5_deploy.md) — S 版完整技術手冊(五階段、GPU OOM 解法、內建 NMS)
  - [260627_hailo-nms-endnodes.md](./260627_hailo-nms-endnodes.md) — 量化 GPU OOM、內建 NMS、end-node 切點
  - [260627_hailo-reid-multinet.md](./260627_hailo-reid-multinet.md) — OSNet→ONNX、MOT20 GT crop、ClientRunner.join 合成單一 HEF
  - [260627_rpi5-hailo-multinet-guide.md](./260627_rpi5-hailo-multinet-guide.md) — network_name 分流、串行兩網路、ReID 批次(可複製到 Pi)
- **匯出對話**: `yolox-n.txt`

---

## 一、故事:做什麼、為什麼(What & Why)

把在 MOT20 fine-tune 的 YOLOX 行人偵測 + ByteTrack 追蹤,量化成 INT8 部署到
**Raspberry Pi 5 + Hailo-8/8L** 加速器。本階段從 S 版延伸到 **nano 版**,並把
**OSNet ReID** 共同編譯成單一多網路 HEF(`bytetrack_yoloxn_osnet.hef`),
讓裝置只 configure 一次。

起點問題鏈(本 session 實際走過):
1. `exps/default` vs `exps/example` 差異 → 釐清 detection-only vs MOT tracking。
2. MOT17 推理算 MOTA → `track.py` 用法、`-b 1` 限制(tracker 有狀態,只取 `outputs[0]`)。
3. `--save_result --eval` 報錯 → 此版 `track.py` 無這些 flag,CLAUDE.md 文件有誤。
4. `--fp16` 推理是否有效 → 有效(`mot_evaluator` 會 `model.half()`)。
5. `KeyError: 'info'` → annotation JSON 缺 `info` 欄位(`train.json` 缺、`val_half.json` 有)。
6. MOT20 推理/fine-tune → symlink 資料集、patch JSON、建 exp。
7. 換 nano + OSNet ReID → 本檔重點。

---

## 二、逐步思路(Step by Step)

### A. 評估設定修正(data leakage + KeyError)
- 根因:`yolox_*_mix_det.py` 的 `val_ann="train.json"` 既洩漏訓練資料、又因缺 `info`
  讓 `pycocotools.loadRes` 崩潰。
- 解法:全部改 `val_ann="val_half.json"`;MOT20 的 JSON 用腳本補 `info={}`。

### B. MOT20 資料接入
- `datasets/MOT20` symlink 指向 Fast-Deep-OC-SORT 的 MOT20。
- `yolox_x_mix_mot20_ch.py` 的 `val_ann` 改 `val_half.json`、eval `name` 改 `train`。

### C. 建 fine-tune exp(strategy A:純 MOT20,不混 CrowdHuman)
- S 版:`yolox_s_mot20_ft.py`;nano 版:`yolox_nano_mot20_ft.py`(本檔)。
- 關鍵:`train_half.json` 訓練 / `val_half.json` 評估(無 frame 重疊);
  `max_epoch=30`、`no_aug_epochs=5`、`lr ÷10`。

### D. Hailo 工具鏈調查(關鍵限制)
- **DFC 只能在 x86_64 跑,RPi5(ARM)無法編譯** → PC 編 HEF,複製到 Pi 跑。
- 管線:`ONNX -> parse(HAR) -> optimize/quantize INT8(HAR,需校準) -> compile(HEF)`。
- `hw_arch`:AI Kit=`hailo8l`(13 TOPS),AI HAT+=`hailo8`(26 TOPS),HEF 綁架構。

### E. nano 可否「免改 code」驗證
- 匯出 nano ONNX 比對:input 名 `images`、608x1088、9 個 end-node 名稱、
  output `[1,13566,6]` 全部與 S 版一致 → 兩支 Hailo 腳本**不需改**。
- 兩個缺口:❶ 需新增 nano MOT20 fine-tune exp(本檔已建);
  ❷ OSNet `.onnx` 不在 repo,需重新從 torchreid 匯出。

---

## 三、Checkpoint(原始碼 / 指令 / log)

### 已驗證指令
```bash
# 匯出 nano ONNX 驗證節點名(通過:9 end-nodes 與 S 版一致)
python3 tools/export_onnx.py --output-name /tmp/nano_check.onnx \
    -f exps/example/mot/yolox_nano_mix_det.py \
    -c pretrained/bytetrack_nano_mot17.pth.tar --no-onnxsim

# 驗證新 exp 建模(通過:0.90M params、depthwise=True、width 0.25)
python3 -c "from yolox.exp import get_exp; \
  exp=get_exp('exps/example/mot/yolox_nano_mot20_ft.py',None); m=exp.get_model()"
```

### nano 完整 workflow(待執行)
```bash
# Phase 1 fine-tune
python3 tools/train.py -f exps/example/mot/yolox_nano_mot20_ft.py \
    -d 1 -b 16 --fp16 -c pretrained/bytetrack_nano_mot17.pth.tar
# Phase 2 export onnx
python3 tools/export_onnx.py --output-name bytetrack_nano_mot20.onnx \
    -f exps/example/mot/yolox_nano_mot20_ft.py \
    -c YOLOX_outputs/yolox_nano_mot20_ft/best_ckpt.pth.tar --no-onnxsim
# Phase 3 OSNet onnx(torchreid, 256x128, input name 'images')
# Phase 4 co-compile joined HEF(PC + DFC)
python3 tools/hailo_compile.py --onnx bytetrack_nano_mot20.onnx \
    --calib-dir datasets/MOT20/train --hw-arch hailo8l \
    --output bytetrack_yoloxn_osnet.hef --finetune-batch 4 \
    --second-onnx osnet_ain_x1_0.onnx --second-calib-dir datasets/reid_crops \
    --second-name osnet_reid
# Phase 5 deploy on Pi
python3 deploy/Hailo/hailo_inference.py -m bytetrack_yoloxn_osnet.hef -i <video> --mot20
```

### 既有產物(repo root)
```
bytetrack_s_mot20.onnx / *_h8l.hef / *_h8l_nms.hef / *_quant.har
bytetrack_yolox_osnet.hef / *_joined_quant.har / *_quant.har   # S 版 + OSNet
pretrained/bytetrack_nano_mot17.pth.tar                         # nano 起點
datasets/reid_crops/*.jpg (4000 張)                            # ReID 校準
```

---

## 四、關鍵檔案 / 路徑 / 函式 / 呼叫路徑

### 本檔 `exps/example/mot/yolox_nano_mot20_ft.py`
- `class Exp(MyExp)`:覆寫 detection config。
- `get_model()`:`YOLOPAFPN(depthwise=True)` + `YOLOXHead(depthwise=True)` → nano 招牌。
- `get_data_loader()`:`MOTDataset(MOT20, train_half) -> MosaicDetection -> DataLoader`。
- `get_eval_loader()`:`MOTDataset(MOT20, val_half)`,`SequentialSampler`(保序給 tracker)。
- `get_evaluator()`:`COCOEvaluator`(AP);MOTA 由 `track.py` 外層 `motmetrics` 算。
- 呼叫者:`train.py`(train)、`track.py`(eval)、`export_onnx.py`(只用 `get_model`)。

### Hailo 腳本(對 nano 不需改)
| 檔案 | 角色 | 關鍵 |
|------|------|------|
| `tools/export_onnx.py` | 匯出 ONNX | 已修 `torch.onnx._export -> export` |
| `tools/hailo_compile.py` | PC 編譯(需 DFC) | 寫死 `INPUT_H,W=608,1088`、`END_NODES`(9 conv)、`START_NODE='images'`;`--second-onnx` 走 `ClientRunner.join(JoinAction.NONE)` 共編 ReID;OSNet global avgpool 需 `global_avgpool_reduction` 拆解避免 INT8 溢位 |
| `deploy/Hailo/hailo_inference.py` | Pi 執行(需 HailoRT) | auto-detect 三種 HEF;`reassemble_yolox()` 把 9 個 NHWC conv 重組 `[1,13566,6]`(obj/cls 補 sigmoid)→ `demo_postprocess` → `multiclass_nms` → `BYTETracker` |

### 正確性錨點
- preproc(`yolox/data/data_augment.py::preproc`):letterbox 608x1088、pad 114、
  `(x/255-mean)/std`(ImageNet)。Hailo 端用 `normalization([123.675,116.28,103.53],
  [58.395,57.12,57.375])` 把正規化烤進晶片,校準圖餵原始 0-255 RGB。
- end-nodes:`/head/{cls,reg,obj}_preds.{0,1,2}/Conv`(切在 sigmoid 前 → host 補 sigmoid)。
- 已驗:host 重組 + decode 與原始 ONNX `output` 誤差 1.2e-7。

---

## 五、待辦 / 風險

- [ ] OSNet `.onnx` 需重新從 torchreid 匯出(缺口❷;repo 只剩 hef/har)。
- [ ] 確認 Pi 實體是 `hailo8` 還是 `hailo8l`(`hailortcli fw-control identify`)。
- [ ] OSNet `--second-input-name`:torchreid 匯出若 input 非 `images` 需指定。
- [ ] DFC <-> HailoRT 版本相容;DFC 安裝需 Hailo Developer Zone 帳號。
- [ ] (選用)Pi 上 dump MOT 格式 track_results,重跑 eval 量 INT8 MOTA 掉幅。
