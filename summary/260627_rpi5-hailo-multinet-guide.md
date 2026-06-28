# RPi5 + Hailo-8L:多網路 HEF(YOLOX-S + OSNet ReID)推理程式指南

> 紀錄日期:2026-06-27
> 目標機:Raspberry Pi 5 + Hailo-8L(13 TOPS,AI Kit M.2)
> HEF:`bytetrack_yolox_osnet.hef`(在 PC 用 `tools/hailo_compile.py` 編出後複製到 Pi)
> 主程式:`deploy/Hailo/hailo_inference.py`
> 相關主軸:`summary/hailo8_rpi5_deploy.md`、`summary/260627_hailo-reid-multinet.md`

本檔是「拿到 HEF 後,在 RPi5 上怎麼跑」的程式指南,可直接複製到 Pi 當 guideline。

---

## 故事 / 背景(What & Why)

ReID-based MOT 是雙模型(YOLOX 偵測 + ReID 抽特徵)。若兩者各自一顆 HEF,
每幀在晶片來回重配(`configure()`)→ context-switch 延遲大。解法:把兩個網路
用 `ClientRunner.join(JoinAction.NONE)` 編進**同一顆多網路 HEF**,device 只
configure 一次、兩網路共住,推理時用 **`network_name`** 選要跑哪個。仍是兩次
`infer`(ReID 相依於偵測 crop),但省掉重配延遲。

這顆 HEF 是 **1 個 network group `joined_yolox_osnet_reid`**,內含 2 個 network:
- `.../yolox`:input 608x1088x3,輸出 9 個 raw conv(本顆未帶內建 NMS)
- `.../osnet_reid`:input 256x128x3,輸出 512 維 embedding(`fc49`)

---

## 一次性:Pi 環境準備

```bash
sudo apt update && sudo apt install hailo-all      # HailoRT + hailo_platform
hailortcli fw-control identify                     # 確認 Device Architecture = HAILO8L
hailortcli parse-hef bytetrack_yolox_osnet.hef     # 看兩個 network 的 I/O
```
> HEF 綁架構:hailo8l 的 HEF 不能載到 hailo8,反之亦然。
> ByteTrack repo 也要在 Pi 上 `python3 setup.py develop`(需要 `yolox` 套件做後處理)。

---

## 核心觀念(務必先懂)

1. **兩網路串行、有資料相依,不同時餵**:
   ```
   整幀 → yolox → boxes → (CPU: decode+crop+resize) → osnet_reid → embeddings
   ```
   餵 image 給 yolox 時,osnet_reid 不輸入任何東西;ReID 的輸入是 boxes 出來後才裁的 crop。
2. **用 `network_name` 選網路**:`InputVStreamParams.make(..., network_name=...)`
   /`OutputVStreamParams.make(..., network_name=...)` 可把 vstream 限定到單一網路,
   於是能「只跑 yolox」或「只跑 osnet_reid」。
3. **ReID 可以 Batch**:整幀所有 crop 疊成 `(N,256,128,3)` 一次 `infer` → `(N,512)`,
   比一個一個快。一次 device activate 內先跑 yolox、再跑 reid。
4. **前處理餵 raw 0-255 RGB**:normalization 已烤進 HEF(YOLOX letterbox 608x1088 pad114;
   ReID 直接 resize 256x128)。**不要**自己除 255 或減 mean。

---

## 程式骨架(已實作於 `deploy/Hailo/hailo_inference.py`)

`HailoPredictor.__init__` 自動偵測:
- 掃 `get_input/output_vstream_infos()` 的 `.network_name` → 找出 reid 網路(名稱含
  `reid`/`osnet`)與 yolox 網路。
- yolox / reid 各自建一組 `network_name` scoped 的 vstream params。
- 偵測 yolox 是 raw-conv 還是內建 NMS(`--builtin-nms auto/on/off`)。
- `--reid auto/on/off` 控制要不要跑 ReID。

每幀 `inference()` 的流程(對應 pseudo code):
```python
with network_group.activate(ng_params):                 # 一次 activate
    # Stage A:偵測(只跑 yolox vstreams)
    with InferVStreams(ng, yolox_in, yolox_out) as p:
        raw = p.infer({yolox_input_name: img[None]})     # img: letterbox 608x1088 0-255 RGB
    dets = parse_builtin_nms(raw) or decode_yolox(raw)   # [N,5] xyxy(px)+score

    # Stage B:ReID(只跑 osnet_reid vstreams;boxes 出來才裁 crop)
    embs = None
    if multinet and dets is not None:
        crops = stack([resize(frame[y1:y2,x1:x2], 256,128)[...,::-1] for box in dets])  # (N,256,128,3)
        with InferVStreams(ng, reid_in, reid_out) as p:
            embs = p.infer({reid_input_name: crops})[reid_output_name]    # (N,512)
        embs = l2_normalize(embs)
return dets, embs, img_info
```

執行:
```bash
python3 deploy/Hailo/hailo_inference.py \
    -m bytetrack_yolox_osnet.hef -i <video.mp4> --mot20
```

---

## Checkpoint(在 Pi 上驗證的東西)

- `hailortcli parse-hef` 應印出 2 個 network、各自 I/O(下方為 PC 端已驗證的內容):
  ```
  Network group: joined_yolox_osnet_reid, Multi Context - 14 contexts
    osnet_reid: IN osnet_reid/input_layer1 UINT8 NHWC(256x128x3) | OUT osnet_reid/fc49 UINT8 NC(512)
    yolox     : IN yolox/input_layer1 UINT8 NHWC(608x1088x3)     | OUT conv54..conv83 (9 個 raw conv)
  ```
- 第一幀程式會印 `[hailo] multinet=True yolox='.../yolox' builtin_nms=False reid='.../osnet_reid'`
  與 `[hailo] ReID embeddings per frame: (N, 512)`。
- **座標 sanity check**:首跑請確認偵測框畫在人身上、ReID crop 是完整行人。

---

## Options / 關鍵檔案與結構

| 檔案 / 符號 | 角色 |
|------|------|
| `deploy/Hailo/hailo_inference.py` | Pi 主程式 |
| `HailoPredictor.__init__` | 用 `network_name` 分流 yolox/reid 的 vstream |
| `HailoPredictor.inference` | 一次 activate 內 Stage A→B 串行 |
| `HailoPredictor.embed` | crop+resize 256x128 → 批次 ReID → L2 normalize |
| `HailoPredictor.decode_yolox` / `parse_builtin_nms` | YOLOX 兩種後處理 |
| `reassemble_yolox` | 9 個 conv → `[1,n_anchors,6]`(補 sigmoid) |

CLI 旗標:`-m -i -o -s(score_thr) -n(nms_thr) --builtin-nms --reid --mot20`
與 tracking 參數(`--track_thresh/buffer/match_thresh/min-box-area`)。

---

## Risk / TODO

- [ ] **embeddings 目前未被使用**:`BYTETracker` 是純 motion,不吃 appearance。
      要真的用 ReID,得把 `embs`(N,512,與 dets 1:1)接到 **BoT-SORT/DeepSORT** 的 `update()`。
      目前程式只算出 embedding 並印 shape,當骨架。
- [ ] **ReID 框座標**:`embed()` 直接用原圖像素 xyxy 裁切;若 YOLOX 用內建 NMS 版,
      座標換算路徑不同,需對齊。
- [ ] **HailoRT batch 行為**:HEF 以 batch=1 編;`infer` 餵 `(N,...)` 由 HailoRT 串流處理。
      N 很大(MOT20 高人數)時注意延遲/記憶體,必要時分塊。
- [ ] **內建 NMS 框順序**隨 HailoRT 版本可能不同(見 `parse_builtin_nms` 註解),首跑對照驗證。
- [ ] **ReID 精度**:HEF 的 ReID 走 optimization_level=0(避開 bias_correction crash),精度略降;
      higher level 修復為 PC 端 TODO(見 `260627_hailo-reid-multinet.md`)。
- [ ] **效能量測**:在 Pi 上量整體 FPS;雙網路 + 高人數時 ReID 批次大小是主要變因。

---

## 參考
- `summary/260627_hailo-reid-multinet.md`(PC 端:OSNet→ONNX、avgpool reduction、join 編譯)
- `summary/260627_hailo-nms-endnodes.md`(YOLOX end-node、內建 NMS、量化 OOM)
- HailoRT:`InputVStreamParams.make(..., network_name=...)`、`HEF.get_*_vstream_infos()`
