# YOLOX-S (ByteTrack/MOT20) 部署到 Raspberry Pi 5 + Hailo-8

> 紀錄日期:2026-06-25
> 來源模型:`YOLOX_outputs/yolox_s_mot20_ft/best_ckpt.pth.tar`
> (YOLOX-S, num_classes=1 person, input 608x1088, val AP@0.5 = 0.561)

## 目標與背景

把在 MOT20 上 fine-tune 的 YOLOX-S 模型,量化成 INT8 後部署到
**Raspberry Pi 5 + Hailo-8 加速器**(AI HAT+ / AI Kit),用 Hailo 自家工具鏈
**Dataflow Compiler (DFC)** 把 ONNX 編譯成 `.hef`(Hailo Executable Format)。
INT8 量化是 DFC 編譯流程內建的一個階段,不是另外用 ONNX Runtime 做。

## 關鍵限制(務必先懂)

### 編譯與執行必須分機
- **DFC 只能在 x86_64 Linux 跑,RPi5 (ARM) 無法編譯。**
  本機開發 PC 符合:x86_64、62GB RAM(>32GB 門檻)、RTX 3070(加速量化)。
  → 在 PC 編出 `.hef`,複製到 RPi5 執行。
- RPi5 只負責用 HailoRT (`hailo-all`) + `hailo_platform` Python API 跑 `.hef`。

### hw_arch 要對應實體硬體
| 硬體 | hw_arch | 算力 |
|------|---------|------|
| 原版 RPi5 AI Kit (M.2) | `hailo8l` | 13 TOPS |
| AI HAT+ | `hailo8` | 26 TOPS |

HEF 綁架構,選錯不能載入。上 Pi 用 `hailortcli fw-control identify` 確認。

## 五階段管線

```
ONNX --parse--> HAR --optimize/quantize(INT8)--> HAR --compile--> HEF
```

### Stage 0 — 安裝工具鏈(一次性)
- **PC 端(編譯)**:DFC 非 PyPI,需從 Hailo Developer Zone(免費帳號)下載
  `hailo_dataflow_compiler-<ver>-...-linux_x86_64.whl`,或用官方
  **Hailo AI Software Suite docker**(建議,內含 DFC + Model Zoo + 相依)。
  DFC 版本要與 Pi 上的 HailoRT 版本相容。
- **RPi5 端(執行)**:
  ```bash
  sudo apt update && sudo apt install hailo-all
  hailortcli fw-control identify        # 確認 device + hailo8 vs 8l
  ```

### Stage 1 — 匯出 ONNX(PC)
重用 `tools/export_onnx.py`(已 `decode_in_inference=False`,輸出原始 head,
grid-decode + NMS 留在晶片外)。
```bash
python3 tools/export_onnx.py \
    --output-name bytetrack_s_mot20.onnx \
    -f exps/example/mot/yolox_s_mot20_ft.py \
    -c YOLOX_outputs/yolox_s_mot20_ft/best_ckpt.pth.tar \
    --no-onnxsim
```
> 修過的 bug:新版 torch 移除 `torch.onnx._export`,已改為 `torch.onnx.export`。

### Stage 2 — 找 end-nodes(PC)
Hailo 切在 9 個 detection-head conv 輸出(3 strides x {cls, reg, obj}):
```
/head/cls_preds.{0,1,2}/Conv
/head/reg_preds.{0,1,2}/Conv
/head/obj_preds.{0,1,2}/Conv
```
切點在 conv(sigmoid 之前),所以 obj/cls 的 sigmoid 要在 host 端補。

### Stage 3+4 — parse/量化/compile + 校準(PC)
腳本:`tools/hailo_compile.py`(`ClientRunner` API)
```bash
python3 tools/hailo_compile.py \
    --onnx bytetrack_s_mot20.onnx \
    --calib-dir datasets/MOT20/train \
    --hw-arch hailo8 \
    --output bytetrack_s_mot20.hef
```
重點:
- **normalization 烤進晶片**:模型腳本
  `normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])`
  正好等於 preproc 的 `(x/255 - mean)/std`(ImageNet mean/std × 255)。
- **校準集餵原始 0-255 RGB**(letterbox 到 608x1088、pad 114),
  不要先除 255 或減 mean(晶片做)。從 MOT20 抽 ~1024 張。
- 量化最吃資源(建議 GPU,純 CPU 要數小時)。

### Stage 5 — RPi5 部署執行
腳本:`deploy/Hailo/hailo_inference.py`(HailoRT + BYTETracker)
```bash
python3 deploy/Hailo/hailo_inference.py \
    -m bytetrack_s_mot20.hef -i <video.mp4> --mot20
```
流程:letterbox 0-255 RGB → HailoRT 推理 → 9 個 NHWC 輸出重組成
`[1, n_anchors, 6]`(obj/cls 補 sigmoid)→ 重用 `demo_postprocess`(grid decode)
+ `multiclass_nms` → `BYTETracker`。

## 正確性驗證(已完成)

用 ONNX 抽出 9 個 conv 輸出、模擬 Hailo 的 NHWC,跑 `reassemble_yolox` +
`demo_postprocess`,結果與原始 ONNX `output` 誤差僅 **1.2e-7**。
證明 host 端解碼(channel 順序 [reg,obj,cls]、sigmoid、scale 排序 stride 8/16/32、
grid decode)完全正確。HailoRT 裝置 I/O glue 因無實機尚未測。

## 相關檔案

| 檔案 | 角色 |
|------|------|
| `tools/export_onnx.py` | Stage 1 匯出(已修 torch API) |
| `tools/hailo_compile.py` | Stage 3-4(PC 編譯,需 DFC) |
| `deploy/Hailo/hailo_inference.py` | Stage 5(RPi5 執行,需 HailoRT) |
| `yolox/data/data_augment.py::preproc` | preproc 基準(letterbox/normalization) |
| `yolox/models/yolo_head.py:98-116` | cls/reg/obj head = end-node 來源 |
| `yolox/utils/demo_utils.py` | `demo_postprocess` + `multiclass_nms` |

## 待辦 / 風險

- [ ] Stage 0:下載安裝 DFC(PC)、`hailo-all`(Pi)
- [ ] 確認 `hailo8` vs `hailo8l`(`hailortcli fw-control identify`)
- [ ] DFC ↔ HailoRT 版本相容
- [ ] 608x1088 為非方形大輸入,compile 若資源分配失敗,考慮 performance flags
- [ ] (選用)在 Pi 上 dump MOT 格式 track_results,重跑 eval 量測 INT8 的 MOTA 掉幅

## 參考來源

- Raspberry Pi — AI Kit Dataflow Compiler now available
- Ultralytics Docs — Hailo Export
- Christian Mills — YOLOX Object Tracking on RPi5 AI Kit
- RidgeRun — Convert ONNX to Hailo8L
- Hailo Model Zoo — GETTING_STARTED
