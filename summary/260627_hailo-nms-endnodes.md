# Hailo 內建 NMS、量化 OOM 解法、與 YOLOX end-node 切點

> 紀錄日期:2026-06-27
> 接續主軸:`summary/hailo8_rpi5_deploy.md`(YOLOX-S/ByteTrack/MOT20 → RPi5 + Hailo-8)
> 本篇聚焦三件事:量化 GPU OOM 解法、on-chip NMS、為何 end-node 切在 9 顆 conv

---

## 故事 / 背景(What & Why)

把 MOT20 fine-tune 的 YOLOX-S(`bytetrack_s_mot20.onnx`,input 608x1088,1 class)
編成 Hailo `.hef` 部署到 **RPi5 + Hailo-8L(13 TOPS)**。本次解掉三個卡點:

1. **量化階段 GPU OOM** — DFC 預設 optimization_level=2 含 QAT fine-tuning,
   608x1088 大輸入把 RTX 3070(8GB)撐爆。
2. **內建 NMS** — 仿 Hailo Model Zoo,把 sigmoid + grid-decode + per-class NMS
   烤進 HEF,簡化 Pi 端程式。
3. **end-node 觀念釐清** — 使用者用 Netron 看到輸出是 `/head/Transpose`,
   但我們切在 9 顆 conv,釐清兩者差異。

---

## 思路 step by step

### A. GPU OOM
- 錯誤:`AccelerasResourceError: GPU memory has been exhausted ...`
  訊息建議「降低 fine-tune batch size 或改 CPU」。
- 根因:1024 張校準圖 → DFC 自動套 optimization_level=2(含 QAT)→ 大輸入爆顯存。
- 對策(保留 GPU、精度最佳):把 fine-tune batch size 調小。**實測 `--finetune-batch 4` 成功**。
- 在 `tools/hailo_compile.py` 加三個旋鈕,注入 alls model script:
  `--finetune-batch`、`--optimization-level`、(搭配 `--num-calib`)。

### B. 內建 NMS(on-chip nms_postprocess)
- 我們的模型就是標準 YOLOX decoupled head → 直接適用 Model Zoo 同款
  `nms_postprocess("<json>", yolox, engine=cpu)`。
- 關鍵:config 用的是 **HAR 內部 layer 名(convNN)**,非 ONNX 名;**而且 convNN 會隨
  模型變(S vs nano 不同)**。
- **改良:`--nms-config auto`(2026-06-28)**——`hailo_compile.py` 改成 parse 後**自動**
  從 HAR 產生 nms config(`build_yolox_nms_config()`:把 9 顆 END_NODE 依
  `original_names` 對回 convNN、依 stride 8/16/32 分組、填 reg/obj/cls)。
  從此**不用每個模型手寫 json**,任何 YOLOX 變體(S、nano-depthwise…)通用。
  搭配 `--nms-classes/--nms-score-th/--nms-iou-th/--nms-max`(預設 1/0.1/0.65/200)。
  產物存成 `<output>_nms_config.json` 可供檢視。已用 S 模型驗證:auto 重現原手寫
  對應(conv54/55/56…)完全一致。手寫 json 路徑仍可用(向後相容)。
- **ByteTrack 專屬注意**:`nms_scores_th` 是 compile-time hardcode,烤進 HEF 後 Pi 改不了。
  設低(0.1)保留第二階段低分框;Pi 端用 `--score_thr`/`--track_thresh` 往上調
  (host 只能再過濾更低分,不能低於烤進去的值)。
- `engine=cpu`:NMS 跑 Pi 的 ARM CPU(HailoRT 內),非 NN core;人多時量延遲。

### C. 為何 end-node 切在 9 顆 conv(不是 `/head/Transpose`)
- 名字來源:`yolo_head.py:42-44` 的 `cls_preds/reg_preds/obj_preds` 三個 ModuleList,
  每 stride 一顆 → 3x3=9 顆。PyTorch 匯出時節點名 = module 屬性路徑 →
  `/head/cls_preds.0/Conv` 等(機械生成,非臆測)。
- forward(`yolo_head.py:159-216`):9 顆 conv → `cat([reg, obj.sigmoid(), cls.sigmoid()])`
  → flatten/concat/`permute(0,2,1)`(= ONNX 的 `/head/Transpose`)→ output `[1,13566,6]`。
  `decode_in_inference=False`(export 設)→ **不在 ONNX 做 grid decode**,留給 host。
- Netron 看到的 `/head/Transpose` **沒錯**,那是 ONNX 真正輸出(已 sigmoid、已 reshape、
  未 grid-decode)。但 Hailo 刻意往前切到 9 顆 raw conv,因為:
  1. conv 之後是 Sigmoid/Reshape/Transpose/Concat + 動態 Shape/Slice/Constant,
     動態 shape 膠水 op 不吃 Hailo 靜態 dataflow。
  2. 9 顆 conv 是重運算終點,之後都便宜、該回 host 或交給 meta-arch。
  3. `nms_postprocess(yolox)` 規定餵 sigmoid 之前的 raw conv(它自己補 sigmoid+grid)。
     → 這也是 host 端 `reassemble_yolox` 要自己 `_sigmoid(obj/cls)` 的原因。

---

## Checkpoint(已驗證的事實 / log)

### HAR layer 對應(YOLOX-S;`--nms-config auto` 現已自動產生,不必手抄)
> 此表是 YOLOX-S 的對應,nano 等變體編號不同 → 用 `--nms-config auto` 自動推導。
| stride | cls | obj | reg |
|--------|-----|-----|-----|
| 8  | conv54 | conv55 | conv56 |
| 16 | conv68 | conv69 | conv70 |
| 32 | conv81 | conv82 | conv83 |

### 多變體編譯(nano 三顆 HEF 範例)
先在 YOLOX 環境匯出 `bytetrack_yoloxn_mot20.onnx`(DFC venv 缺 loguru/torchvision),再:
```bash
# ① standalone raw-conv
python3 tools/hailo_compile.py --onnx bytetrack_yoloxn_mot20.onnx \
    --calib-dir datasets/MOT20/train --hw-arch hailo8l \
    --output bytetrack_yoloxn.hef --finetune-batch 4
# ② 內建 NMS（auto config）
python3 tools/hailo_compile.py --onnx bytetrack_yoloxn_mot20.onnx \
    --calib-dir datasets/MOT20/train --hw-arch hailo8l \
    --output bytetrack_yoloxn_nms.hef --finetune-batch 4 --nms-config auto
# ③ 多網路 + ReID（可再加 --nms-config auto）
python3 tools/hailo_compile.py --onnx bytetrack_yoloxn_mot20.onnx \
    --calib-dir datasets/MOT20/train --hw-arch hailo8l \
    --output bytetrack_yoloxn_osnet.hef --finetune-batch 4 \
    --second-onnx .../osnet_ain_x1_0.onnx --second-calib-dir datasets/reid_crops
```

### ONNX 結構(`onnx` 掃圖)
- 輸入:`images [1,3,608,1088]`;輸出:`output [1,13566,6]`(6 = reg4+obj1+cls1)。
- anchor 數驗算:`76x136 + 38x68 + 19x34 = 10336+2584+646 = 13566` ✓
- 末端鏈:`9 convs → Sigmoid → Concat/Reshape → /head/Concat_6 → /head/Transpose → output`

### 成功指令(GPU 量化,batch 4)
```bash
python3 tools/hailo_compile.py \
    --onnx bytetrack_s_mot20.onnx --calib-dir datasets/MOT20/train \
    --hw-arch hailo8l --output bytetrack_s_mot20_h8l.hef --finetune-batch 4
```

### 下一步:編內建 NMS 版 HEF
```bash
python3 tools/hailo_compile.py \
    --onnx bytetrack_s_mot20.onnx --calib-dir datasets/MOT20/train \
    --hw-arch hailo8l --output bytetrack_s_mot20_h8l_nms.hef \
    --finetune-batch 4 \
    --nms-config deploy/Hailo/nms_config_bytetrack_s_mot20.json
# 編後驗證:hailortcli parse-hef bytetrack_s_mot20_h8l_nms.hef  → 應見 1 個 HAILO_NMS 輸出
```

### 環境
- DFC `hailo_sdk_client` 3.29.0(已裝,x86_64 PC);GPU RTX 3070(8GB)。
- Model Zoo:`/local/workspace/hailo_model_zoo`(yolox alls/nms config 範例)。

---

## Options / 關鍵檔案與結構

| 檔案 / 符號 | 角色 |
|------|------|
| `tools/hailo_compile.py` | PC 編譯;新增 `--finetune-batch` `--optimization-level` `--nms-config` |
| `deploy/Hailo/nms_config_bytetrack_s_mot20.json` | 內建 NMS config(classes=1, 608x1088, scores_th=0.1, max=200) |
| `deploy/Hailo/hailo_inference.py` | Pi 執行;新增 `--builtin-nms auto/on/off` + `parse_builtin_nms()` + 自動偵測 |
| `yolox/models/yolo_head.py:42-44` | cls/reg/obj_preds ModuleList = 9 conv end-node 來源 |
| `yolox/models/yolo_head.py:159-216` | head forward;sigmoid 位置、`/head/Transpose`、decode_in_inference |
| `yolox/utils/demo_utils.py` | `demo_postprocess`(grid decode)+ `multiclass_nms`(raw-conv 路徑) |

### Call path(Pi 端兩條路徑,自動偵測)
- raw-conv:`HailoRT.infer → reassemble_yolox(+sigmoid) → demo_postprocess → multiclass_nms → BYTETracker`
- 內建 NMS:`HailoRT.infer → parse_builtin_nms → BYTETracker`

### Risk / TODO
- [ ] 編內建 NMS 版 HEF(上方指令);若 OOM 再降 `--finetune-batch 2`。
- [ ] 內建 NMS 框座標**順序/正規化**隨 HailoRT 版本可能不同
      (現按 `(y1,x1,y2,x2)` 正規化到輸入尺寸實作)。首跑拿 raw-conv HEF 對照驗證。
- [ ] `engine=cpu` NMS 延遲在 MOT20 高密度場景需實測。
- [ ] 兩種 HEF 都保留:`..._nms.hef` 當主線、raw-conv 版當驗證基準。
- [ ] 上 Pi 確認 `hailortcli fw-control identify` 為 HAILO8L(HEF 綁架構)。
- [ ] (選用)Pi 上 dump MOT track_results 重跑 eval,量 INT8 的 MOTA 掉幅。

---

## 參考
- Hailo Model Zoo:`cfg/alls/generic/yolox_*.alls`、`cfg/postprocess_config/nms_config_yolox_*.json`
- `https://github.com/hailo-ai/hailo_model_zoo`
- `docs/public_models/HAILO8L/HAILO8L_object_detection.rst`(YOLOX 在 8L 用內建 NMS)
