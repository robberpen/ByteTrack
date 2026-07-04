#!/usr/bin/env python3
# encoding: utf-8
# Compile a YOLOX-S (ByteTrack/MOT) ONNX model into a Hailo HEF.
#
# RUNS ON THE x86_64 PC (NOT on the Raspberry Pi). Requires the Hailo Dataflow
# Compiler (hailo_sdk_client), which is downloaded from the Hailo Developer Zone.
#
# Pipeline:  ONNX --parse--> HAR --optimize/quantize(INT8)--> HAR --compile--> HEF
#
# The 9 detection-head conv outputs are used as end nodes; YOLOX grid-decode + NMS
# stay OFF-chip (done on the Pi by deploy/Hailo/hailo_inference.py). Input
# normalization is baked ON-chip via a model script, so the calibration images are
# fed as raw 0-255 RGB (letterboxed, pad 114) -- identical to what the Pi will feed.
#
# Example:
#   python3 tools/hailo_compile.py \
#       --onnx bytetrack_s_mot20.onnx \
#       --calib-dir datasets/MOT20/train \
#       --hw-arch hailo8 \
#       --output bytetrack_s_mot20.hef
#
# Optional: co-compile a 2nd ReID net (OSNet) into ONE multi-network HEF so the
# Pi configures the device once (no per-frame reconfigure between det and ReID):
#   python3 tools/hailo_compile.py \
#       --onnx bytetrack_s_mot20.onnx --calib-dir datasets/MOT20/train \
#       --hw-arch hailo8l --output bytetrack_yolox_osnet.hef --finetune-batch 4 \
#       --second-onnx osnet_ain_x1_0.onnx --second-calib-dir datasets/reid_crops
# Each net is parsed+quantized independently (own calib) then merged via
# ClientRunner.join(JoinAction.NONE). They stay two networks (two inferences on
# the Pi) -- join only removes the device reconfigure latency, not the data
# dependency (ReID still consumes detector crops).

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np

# ByteTrack/YOLOX preprocessing constants (yolox/data/data_augment.py::preproc and
# deploy/ONNXRuntime/onnx_inference.py). The model was trained with ImageNet
# normalization on a 0-1 scaled image:  x_norm = (x/255 - mean) / std.
INPUT_H, INPUT_W = 608, 1088
RGB_MEANS = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
PAD_VALUE = 114.0

# The 9 YOLOX decoupled-head conv end nodes (3 strides x {cls, reg, obj}).
END_NODES = [
    "/head/cls_preds.0/Conv", "/head/reg_preds.0/Conv", "/head/obj_preds.0/Conv",
    "/head/cls_preds.1/Conv", "/head/reg_preds.1/Conv", "/head/obj_preds.1/Conv",
    "/head/cls_preds.2/Conv", "/head/reg_preds.2/Conv", "/head/obj_preds.2/Conv",
]
START_NODE = "images"
STRIDES = [8, 16, 32]   # YOLOX FPN strides, k=0,1,2 -- for auto nms-config decoders


def make_parser():
    parser = argparse.ArgumentParser("YOLOX -> Hailo HEF compiler")
    parser.add_argument("--onnx", default="bytetrack_s_mot20.onnx",
                        help="input ONNX model exported by tools/export_onnx.py")
    parser.add_argument("--calib-dir", default="datasets/MOT20/train",
                        help="root dir holding <seq>/img1/*.jpg frames for calibration")
    parser.add_argument("--num-calib", type=int, default=1024,
                        help="number of calibration frames to sample")
    parser.add_argument("--hw-arch", default="hailo8", choices=["hailo8", "hailo8l"],
                        help="Hailo target. AI HAT+ (26 TOPS)=hailo8, AI Kit (13 TOPS)=hailo8l")
    parser.add_argument("--output", default="bytetrack_s_mot20.hef",
                        help="output HEF path")
    parser.add_argument("--model-name", default="bytetrack_s_mot20",
                        help="network name inside the HAR")
    parser.add_argument("--seed", type=int, default=0)
    # --- GPU-memory knobs for runner.optimize (quantization) ---
    parser.add_argument("--optimization-level", type=int, default=None,
                        choices=[0, 1, 2, 3, 4],
                        help="Hailo model-optimization level. Lower = less GPU RAM. "
                             "0/1 skip Quantization-Aware Fine-Tuning entirely. "
                             "Default: let DFC pick from --num-calib.")
    parser.add_argument("--finetune-batch", type=int, default=None,
                        help="batch size for Quantization-Aware Fine-Tuning. Lower "
                             "this (e.g. 4 or 2) if the GPU runs out of memory.")
    parser.add_argument("--equalization", action="store_true",
                        help="enable cross-layer equalization (pre-quant). Helps "
                             "depthwise/tiny models (YOLOX nano/tiny) whose per-channel "
                             "weight ranges are hard to quantize.")
    parser.add_argument("--head-16bit", action="store_true",
                        help="quantize the 9 YOLOX head convs at 16-bit (a16_w16) instead "
                             "of 8-bit. Recovers accuracy on quantization-sensitive "
                             "depthwise models (nano) at a small speed cost.")
    parser.add_argument("--layers-16bit", default=None,
                        help="comma-separated extra HAR layer names to force a16_w16 "
                             "(e.g. 'conv40,conv44'). Combined with --head-16bit.")
    # --- on-chip (HailoRT) NMS post-process ---
    parser.add_argument("--nms-config", default=None,
                        help="enable on-chip YOLOX NMS. Use 'auto' to AUTO-generate the "
                             "config from the parsed HAR (maps the 9 head convs per "
                             "stride -- works for any YOLOX variant incl. nano), or pass "
                             "a path to a hand-written json. When set, sigmoid + grid-decode "
                             "+ per-class NMS are baked in and the HEF outputs final boxes.")
    parser.add_argument("--nms-classes", type=int, default=1,
                        help="number of classes for auto nms-config (person-only = 1)")
    parser.add_argument("--nms-score-th", type=float, default=0.1,
                        help="baked-in nms score threshold (keep low for ByteTrack)")
    parser.add_argument("--nms-iou-th", type=float, default=0.65)
    parser.add_argument("--nms-max", type=int, default=200,
                        help="max proposals per class")
    # --- optional 2nd network (ReID) -> single multi-network HEF via ClientRunner.join ---
    parser.add_argument("--second-onnx", default=None,
                        help="optional ReID ONNX to co-compile with YOLOX into ONE "
                             "multi-network HEF (e.g. the OSNet osnet_ain_x1_0.onnx). "
                             "Each net is quantized independently then joined "
                             "(JoinAction.NONE) so the device configures once on the Pi.")
    parser.add_argument("--second-name", default="osnet_reid",
                        help="network + scope name for the ReID net inside the joined HEF")
    parser.add_argument("--second-input-h", type=int, default=256)
    parser.add_argument("--second-input-w", type=int, default=128)
    parser.add_argument("--second-input-name", default="images",
                        help="input tensor name of the ReID ONNX")
    parser.add_argument("--second-calib-dir", default=None,
                        help="dir of person-crop images (*.jpg/*.png, recursive) used to "
                             "calibrate the ReID net. Required when --second-onnx is set.")
    parser.add_argument("--second-num-calib", type=int, default=1024)
    parser.add_argument("--second-optimization-level", type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help="model-optimization level for the ReID net (separate from "
                             "YOLOX). Default 0: verified to quantize OSNet cleanly. "
                             "Higher levels currently crash in bias_correction "
                             "(zero-size array) for OSNet-AIN -- see summary.")
    parser.add_argument("--first-scope", default="yolox",
                        help="scope name for the YOLOX net inside the joined HEF")
    return parser


def letterbox_raw(image, out_h=INPUT_H, out_w=INPUT_W):
    """Letterbox to (out_h, out_w) with pad 114, return RGB float32 in 0-255 (HWC).

    Matches yolox preproc geometry but STOPS before /255 and mean/std, which are
    applied on-chip by the Hailo normalization model script.
    """
    padded = np.ones((out_h, out_w, 3), dtype=np.float32) * PAD_VALUE
    r = min(out_h / image.shape[0], out_w / image.shape[1])
    rh, rw = int(image.shape[0] * r), int(image.shape[1] * r)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded[:rh, :rw] = resized
    padded = padded[:, :, ::-1]  # BGR (cv2) -> RGB
    return np.ascontiguousarray(padded, dtype=np.float32)


def build_calib_set(calib_dir, num_calib, seed):
    """Sample frames and stack into (N, H, W, 3) float32 RGB 0-255 for runner.optimize."""
    random.seed(seed)
    files = sorted(glob.glob(os.path.join(calib_dir, "*", "img1", "*.jpg")))
    if not files:
        raise FileNotFoundError(f"no calibration frames under {calib_dir}/*/img1/*.jpg")
    if len(files) > num_calib:
        files = random.sample(files, num_calib)
    print(f"[calib] using {len(files)} frames from {calib_dir}")

    calib = np.zeros((len(files), INPUT_H, INPUT_W, 3), dtype=np.float32)
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            raise RuntimeError(f"failed to read {f}")
        calib[i] = letterbox_raw(img)
        if (i + 1) % 100 == 0:
            print(f"[calib] {i + 1}/{len(files)}")
    return calib


def build_crop_calib(calib_dir, out_h, out_w, num_calib, seed):
    """Sample ReID person-crop images -> (N, out_h, out_w, 3) float32 RGB 0-255.

    Plain resize (no letterbox/pad) to match the embedder's crop pipeline. Pixels
    stay raw 0-255; ImageNet normalization is baked on-chip by a model script.
    """
    random.seed(seed)
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files += glob.glob(os.path.join(calib_dir, "**", ext), recursive=True)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"no ReID crop images under {calib_dir} (*.jpg/*.png ...)")
    if len(files) > num_calib:
        files = random.sample(files, num_calib)
    print(f"[reid-calib] using {len(files)} crops from {calib_dir}")

    calib = np.zeros((len(files), out_h, out_w, 3), dtype=np.float32)
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            raise RuntimeError(f"failed to read {f}")
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)  # (W,H)
        calib[i] = img[:, :, ::-1].astype(np.float32)  # BGR -> RGB, raw 0-255
        if (i + 1) % 100 == 0:
            print(f"[reid-calib] {i + 1}/{len(files)}")
    return calib


def build_yolox_nms_config(runner, out_path, args):
    """Auto-build the YOLOX nms_postprocess json from the parsed HAR.

    Maps each of the 9 head-conv END_NODES (ONNX names /head/{cls,reg,obj}_preds.K)
    to its HAR layer name (e.g. conv54) via original_names, groups by stride
    8/16/32, and writes the meta-arch config. Works for any YOLOX variant (S, nano,
    ...) because it reads the actual conv names instead of hard-coding them.
    """
    hn = runner.get_hn()
    role_k = {}   # (role, k) -> bare HAR conv name
    for name, ld in hn["layers"].items():
        if ld.get("type") != "conv":
            continue
        for orig in ld.get("original_names", []) or []:
            for role in ("cls_preds", "reg_preds", "obj_preds"):
                for k in range(len(STRIDES)):
                    if orig == f"/head/{role}.{k}/Conv":
                        role_k[(role, k)] = name.split("/")[-1]
    decoders = []
    for k, stride in enumerate(STRIDES):
        missing = [r for r in ("reg_preds", "obj_preds", "cls_preds") if (r, k) not in role_k]
        if missing:
            raise RuntimeError(f"auto nms-config: missing {missing} for stride {stride}")
        decoders.append({
            "name": f"bbox_decoder_{stride}",
            "stride": stride,
            "reg_layer": role_k[("reg_preds", k)],
            "objectness_layer": role_k[("obj_preds", k)],
            "cls_layer": role_k[("cls_preds", k)],
        })
    cfg = {
        "nms_scores_th": args.nms_score_th,
        "nms_iou_th": args.nms_iou_th,
        "number_of_detection_heads": len(STRIDES),
        "image_dims": [INPUT_H, INPUT_W],
        "max_proposals_per_class": args.nms_max,
        "classes": args.nms_classes,
        "bbox_decoders": decoders,
    }
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return out_path


def build_reid_runner(args, mean255, std255):
    """Parse + INT8-quantize the ReID ONNX into its own quantized runner (to join)."""
    from hailo_sdk_client import ClientRunner
    if args.second_calib_dir is None:
        raise SystemExit("--second-onnx requires --second-calib-dir (person crops)")

    r2 = ClientRunner(hw_arch=args.hw_arch)
    print(f"[multinet] parse ReID {args.second_onnx} -> HAR ({args.second_name})")
    # Let DFC auto-detect the (single) end node; just pin the input shape to batch 1.
    r2.translate_onnx_model(
        args.second_onnx, args.second_name,
        net_input_shapes={args.second_input_name:
                          [1, 3, args.second_input_h, args.second_input_w]},
    )
    # OSNet (torchreid) was trained with ImageNet normalization -> bake the same
    # mean/std as YOLOX (ImageNet x255). Distinct var name avoids any clash on join.
    alls2 = "normalization_reid = normalization({}, {})\n".format(mean255, std255)
    # OSNet's many global-avgpool aggregation gates overflow the INT8 accumulator
    # ("shift delta > 2, cannot quantize"). Split each GLOBAL avgpool (output 1x1)
    # into two stages via global_avgpool_reduction(division_factors=[H,1]) -- the
    # same scheme DFC auto-applies to oversized pools. Downsampling avgpools (output
    # > 1x1, e.g. OSNet transitions) are left untouched.
    hn = r2.get_hn()
    n_reduced = 0
    for name, ld in hn["layers"].items():
        if ld.get("type") != "avgpool":
            continue
        ins = ld.get("input_shapes", [[None]])[0]
        outs = ld.get("output_shapes", [[None]])[0]
        if len(outs) >= 3 and outs[1] == 1 and outs[2] == 1:   # global pool
            short, in_h = name.split("/")[-1], ins[1]
            alls2 += ("pre_quantization_optimization(global_avgpool_reduction, "
                      "layers={}, division_factors=[{}, 1])\n".format(short, in_h))
            n_reduced += 1
    print(f"[multinet] global_avgpool_reduction applied to {n_reduced} ReID avgpools")
    # ReID uses its own optimization level (default 0) -- higher levels currently
    # crash OSNet-AIN in bias_correction. Independent of YOLOX's level/finetune.
    alls2 += "model_optimization_flavor(optimization_level={})\n".format(
        args.second_optimization_level)
    r2.load_model_script(alls2)

    calib2 = build_crop_calib(args.second_calib_dir, args.second_input_h,
                              args.second_input_w, args.second_num_calib, args.seed)
    print("[multinet] INT8-quantizing ReID...")
    r2.optimize(calib2)
    return r2


def main():
    args = make_parser().parse_args()

    # Imported here so the calibration/letterbox helpers can be unit-tested without DFC.
    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(hw_arch=args.hw_arch)

    # --- 1. PARSE: ONNX -> HAR, cutting at the 9 head convs ---
    print(f"[parse] {args.onnx} -> HAR ({args.hw_arch})")
    runner.translate_onnx_model(
        args.onnx,
        args.model_name,
        start_node_names=[START_NODE],
        end_node_names=END_NODES,
        net_input_shapes={START_NODE: [1, 3, INPUT_H, INPUT_W]},
    )

    # --- 2. MODEL SCRIPT: bake on-chip normalization ---
    # Hailo normalization computes (x - mean) / std on the raw 0-255 input. To
    # reproduce (x/255 - m)/s we use mean = m*255, std = s*255 (== ImageNet 0-255).
    mean255 = [c * 255.0 for c in RGB_MEANS]   # [123.675, 116.28, 103.53]
    std255 = [c * 255.0 for c in RGB_STD]      # [58.395, 57.12, 57.375]
    alls = "normalization1 = normalization({}, {})\n".format(mean255, std255)
    # Optional GPU-memory controls for the quantization/fine-tuning stage.
    if args.optimization_level is not None:
        alls += "model_optimization_flavor(optimization_level={})\n".format(
            args.optimization_level)
    if args.equalization:
        alls += "pre_quantization_optimization(equalization, policy=enabled)\n"
    # Force selected layers to 16-bit (a16_w16) -- recovers accuracy on layers the
    # 8-bit range can't capture (e.g. nano's depthwise/head). runner is already
    # translated here, so original_names are available to locate the head convs.
    sixteen = []
    if args.head_16bit:
        for name, ld in runner.get_hn()["layers"].items():
            if ld.get("type") != "conv":
                continue
            origs = ld.get("original_names") or []
            if any("/head/{}.".format(r) in (o or "")
                   for o in origs for r in ("cls_preds", "reg_preds", "obj_preds")):
                sixteen.append(name.split("/")[-1])
    if args.layers_16bit:
        sixteen += [s.strip() for s in args.layers_16bit.split(",") if s.strip()]
    if sixteen:
        print("[16bit] a16_w16 layers:", sixteen)
        alls += "quantization_param([{}], precision_mode=a16_w16)\n".format(
            ", ".join(sixteen))
    if args.finetune_batch is not None:
        alls += ("post_quantization_optimization(finetune, policy=enabled, "
                 "batch_size={})\n".format(args.finetune_batch))
    # On-chip YOLOX NMS: sigmoid + grid-decode + per-class NMS baked into the HEF.
    # The config json maps the 3 strides to our HAR conv layers (cls/obj/reg).
    if args.nms_config is not None:
        nms_json = args.nms_config
        if args.nms_config == "auto":
            nms_json = os.path.splitext(args.output)[0] + "_nms_config.json"
            build_yolox_nms_config(runner, nms_json, args)
            print(f"[nms] auto-generated {nms_json}")
        alls += 'nms_postprocess("{}", yolox, engine=cpu)\n'.format(nms_json)
    runner.load_model_script(alls)

    # --- 3. OPTIMIZE / QUANTIZE (INT8) with MOT20 calibration ---
    calib = build_calib_set(args.calib_dir, args.num_calib, args.seed)
    print("[optimize] running INT8 quantization (GPU strongly recommended)...")
    runner.optimize(calib)
    quant_har = os.path.splitext(args.output)[0] + "_quant.har"
    runner.save_har(quant_har)
    print(f"[optimize] saved {quant_har}")

    # --- 3b. OPTIONAL: build + join a 2nd (ReID) network into one HEF ---
    if args.second_onnx is not None:
        from hailo_sdk_client import JoinAction
        r2 = build_reid_runner(args, mean255, std255)
        print(f"[multinet] join '{args.first_scope}' + '{args.second_name}' "
              "(JoinAction.NONE: independent nets, one HEF)")
        runner.join(r2, scope1_name=args.first_scope, scope2_name=args.second_name,
                    join_action=JoinAction.NONE)
        joined_har = os.path.splitext(args.output)[0] + "_joined_quant.har"
        runner.save_har(joined_har)
        print(f"[multinet] saved {joined_har}")

    # --- 4. COMPILE -> HEF ---
    print("[compile] generating HEF...")
    hef = runner.compile()
    with open(args.output, "wb") as f:
        f.write(hef)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
