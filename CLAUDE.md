# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ByteTrack is a multi-object tracking (MOT) framework that achieves state-of-the-art performance by associating every detection box (including low-confidence ones) rather than only high-scoring detections. Built on top of YOLOX object detector, it uses Kalman filtering and IoU-based association for robust tracking across video frames.

**Key Achievement:** 80.3 MOTA, 77.3 IDF1 on MOT17 test set with 30 FPS on V100 GPU.

## Installation and Setup

### Initial Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# Build and install ByteTrack package (creates yolox package)
python3 setup.py develop

# Install additional dependencies
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

**Critical Dependency Note:** The `filterpy` package is required for SORT tracker but may not be automatically installed. If you encounter `ModuleNotFoundError: No module named 'filterpy'`, install it manually:
```bash
pip3 install filterpy
```

### Environment Requirements
- **Python:** 3.6+
- **PyTorch:** 1.7+ (requires >= 1.3, with C++ extensions compiled during setup)
- **CUDA:** Required for training/tracking (GPU strongly recommended)

## Dataset Structure

ByteTrack expects datasets in COCO format under `datasets/`:

```
datasets/
├── mot/
│   ├── train/          # MOT17 training sequences
│   └── test/           # MOT17 test sequences
├── MOT20/
│   ├── train/
│   └── test/
├── crowdhuman/
│   ├── Crowdhuman_train/
│   ├── Crowdhuman_val/
│   ├── annotation_train.odgt
│   └── annotation_val.odgt
├── Cityscapes/
└── ETHZ/
```

### Dataset Conversion
```bash
# Convert datasets to COCO format
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py

# Mix datasets for training
python3 tools/mix_data_ablation.py
python3 tools/mix_data_test_mot17.py
python3 tools/mix_data_test_mot20.py
```

## Training

### Basic Training Command
```bash
python3 tools/train.py \
    -f <exp_file.py> \
    -d <num_gpus> \
    -b <batch_size> \
    --fp16 \
    -o \
    -c <pretrained_checkpoint.pth>
```

**Parameters:**
- `-f`: Experiment config file (see `exps/example/mot/`)
- `-d`: Number of GPUs for distributed training
- `-b`: Total batch size across all GPUs
- `--fp16`: Enable mixed precision training
- `-o`: Occupy GPU memory first
- `-c`: Pretrained checkpoint path

### Training Examples

**Ablation model (MOT17 half + CrowdHuman):**
```bash
python3 tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

**MOT17 test model (full training data):**
```bash
python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

**MOT20 test model:**
For MOT20, bounding boxes must be clipped to image boundaries. Modify:
- `yolox/data/data_augment.py` line 134-135
- `yolox/data/datasets/mosaicdetection.py` lines 122-125, 217-225
- `yolox/utils/boxes.py` lines 115-118

```bash
python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

## Tracking and Evaluation

### Track and Evaluate on MOT17 Half Val
```bash
python3 tools/track.py \
    -f exps/example/mot/yolox_x_ablation.py \
    -c pretrained/bytetrack_ablation.pth.tar \
    -b 1 \
    -d 1 \
    --fp16 \
    --fuse \
    --save_result \
    --eval
```

**Tracking Parameters:**
- `--track_thresh`: High confidence threshold (default: 0.6)
- `--track_buffer`: Frames to keep lost tracks (default: 30)
- `--match_thresh`: IoU threshold for matching (default: 0.9)
- `--min-box-area`: Filter tiny boxes (default: 100)
- `--mot20`: Enable MOT20-specific settings
- `--fuse`: Fuse conv and batch norm layers for faster inference
- `--save_result`: Save tracking results to txt files
- `--eval`: Run evaluation metrics

### Test on MOT17/MOT20 Test Set
```bash
# MOT17
python3 tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py

# MOT20 (with custom input sizes per sequence)
python3 tools/track.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -c pretrained/bytetrack_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --match_thresh 0.7 --mot20
python3 tools/interpolation.py
```

Submit generated txt files to MOTChallenge website for evaluation.

### Compare Different Trackers
```bash
# SORT tracker
python3 tools/track_sort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse

# DeepSORT tracker
python3 tools/track_deepsort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse

# MOTDT tracker
python3 tools/track_motdt.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
```

## Demo and Visualization

```bash
python3 tools/demo_track.py video \
    -f exps/example/mot/yolox_x_mix_det.py \
    -c pretrained/bytetrack_x_mot17.pth.tar \
    --fp16 \
    --fuse \
    --save_result
```

## Architecture

### Core Components

1. **Experiment Configuration System** (`exps/`)
   - All experiments inherit from `BaseExp` (`yolox/exp/base_exp.py`)
   - Experiment files define model architecture, dataset paths, hyperparameters
   - Example: `exps/example/mot/yolox_s_mix_det.py` configures YOLOX-S for MOT
   - Key methods: `get_model()`, `get_data_loader()`, `get_eval_loader()`, `get_optimizer()`

2. **Tracking Modules** (`yolox/tracker/`, `yolox/sort_tracker/`, `yolox/deepsort_tracker/`, `yolox/motdt_tracker/`)
   - **BYTETracker** (`yolox/tracker/byte_tracker.py`): Main ByteTrack algorithm
     - Two-stage association: high-confidence then low-confidence detections
     - Uses Kalman filter for motion prediction
     - Key methods: `update(output_results, img_info, img_size)`
   - **SORT** (`yolox/sort_tracker/sort.py`): Simple online tracker (requires `filterpy`)
   - **DeepSORT** (`yolox/deepsort_tracker/`): Adds appearance features
   - **MOTDT** (`yolox/motdt_tracker/`): Multi-object tracker with detection
   - Shared components:
     - `basetrack.py`: Base track class with state management
     - `kalman_filter.py`: Kalman filter for motion prediction
     - `matching.py`: IoU distance and cost matrix computation

3. **Evaluators** (`yolox/evaluators/`)
   - `mot_evaluator.py`: Imports all trackers (BYTETracker, Sort, DeepSort, MOTDT)
   - **Important:** This is where tracker instantiation happens, so all tracker dependencies must be available
   - Computes MOTA, IDF1, HOTA metrics using `motmetrics` library

4. **Model Architectures** (`yolox/models/`)
   - YOLOX detector variants: nano, tiny, s, m, l, x
   - Custom C++ extensions in `yolox/layers/csrc/` (compiled during setup)

5. **Data Pipeline** (`yolox/data/`)
   - `MOTDataset`: Loads MOT format data converted to COCO
   - `MosaicDetection`: Data augmentation wrapper
   - Training/validation transforms

### Key Design Patterns

- **Experiment-driven architecture:** All configuration through Python experiment files (not YAML/JSON)
- **Tracker interface:** All trackers implement `update(output_results, img_info, img_size)` method
- **Two-stage matching:** ByteTrack's core innovation - associate high-confidence detections first, then recover low-confidence true objects
- **Modular tracker comparison:** Easy to swap trackers via different `tools/track_*.py` scripts

## Common Issues

### Missing Dependencies
- **filterpy:** Required for SORT tracker. Error appears when importing `from filterpy.kalman import KalmanFilter` in `yolox/sort_tracker/sort.py:20`
- **lap:** Required for linear assignment in data association
- **motmetrics:** Required for MOT evaluation metrics

### PYTHONPATH Issues
- The `setup.py develop` installs `yolox` package in development mode
- Imports like `from yolox.evaluators import MOTEvaluator` use the installed package, not relative imports
- Ensure ByteTrack is properly installed via `python3 setup.py develop`

### MOT20 Bounding Box Clipping
- MOT20 requires bounding boxes to be clipped to image boundaries
- Must manually modify specific lines in data augmentation code before training

## Model Zoo

Pre-trained models available from Google Drive and Baidu:
- **Ablation models:** Trained on CrowdHuman + MOT17 half
- **MOT17 models:** bytetrack_{nano,tiny,s,m,l,x}_mot17.pth.tar
- **MOT20 models:** bytetrack_x_mot20.pth.tar
- YOLOX pretrained models: Available from YOLOX model zoo

Place downloaded models in `pretrained/` directory.

## Output Structure

- **Training outputs:** `YOLOX_outputs/<exp_name>/`
- **Tracking results:** `YOLOX_outputs/<exp_name>/track_results/*.txt`
- **Format:** MOT Challenge format (frame, id, x1, y1, w, h, score, -1, -1, -1)
