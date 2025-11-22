#!/usr/bin/env python3
"""
Visualize ID Switches in MOT Tracking Results

This script visualizes tracking results with ID switches highlighted.
It generates annotated video frames showing:
- All tracked bounding boxes with their IDs
- Special highlighting for boxes that experienced ID switches
- Warning markers and text annotations for switch events

Usage Examples:
    # Basic usage with defaults
    python3 tools/visualize_id_switches.py

    # Specify custom tracking results folder
    python3 tools/visualize_id_switches.py -r YOLOX_outputs/yolox_x_ablation/track_results

    # Custom output directory
    python3 tools/visualize_id_switches.py -o visual_switches_custom

    # Generate images only (no video)
    python3 tools/visualize_id_switches.py --no-video

    # Custom video settings
    python3 tools/visualize_id_switches.py --fps 30 --size 1280 720
"""

import os
import sys
import json
import cv2
import glob
import numpy as np
import argparse
from loguru import logger
from pathlib import Path
from collections import defaultdict

# Import analysis functions from analyze_id_switches module
from analyze_id_switches import (
    load_tracking_data,
    create_accumulators,
    extract_id_switches
)


def colormap(rgb=False):
    """
    Get colormap for consistent track ID visualization.
    Same as txt2video.py for compatibility.

    Args:
        rgb: If True, return RGB colors; otherwise BGR for OpenCV

    Returns:
        np.ndarray: Color list of shape (N, 3) with values in [0, 255]
    """
    color_list = np.array([
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def load_coco_annotations(json_path, gt_folder):
    """
    Load COCO format annotations to map frame IDs to image paths.

    Args:
        json_path: Path to COCO JSON file
        gt_folder: Ground truth folder (e.g., 'datasets/mot/train')

    Returns:
        dict: {sequence_name: {frame_id: image_path}}
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    img_dict = defaultdict(dict)

    for img_info in coco_data['images']:
        file_name = img_info['file_name']
        seq_name = file_name.split('/')[0]
        frame_id = img_info['frame_id']
        img_path = os.path.join(gt_folder, file_name)
        img_dict[seq_name][frame_id] = img_path

    return img_dict


def parse_tracking_results(txt_path):
    """
    Parse MOT tracking result file into frame-indexed structure.

    Args:
        txt_path: Path to tracking result txt file

    Returns:
        dict: {frame_id: [(x1, y1, x2, y2, tracker_id, conf), ...]}
    """
    detections = defaultdict(list)

    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            frame_id = int(parts[0])
            tracker_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])

            x2 = x1 + w
            y2 = y1 + h

            detections[frame_id].append((x1, y1, x2, y2, tracker_id, conf))

    return detections


def get_tracker_history(accumulators):
    """
    Build tracker ID history for each ground truth object.
    This tracks which tracker IDs were assigned to each GT object over time.

    Args:
        accumulators: OrderedDict of MOTAccumulators

    Returns:
        dict: {seq_name: {frame_id: {gt_obj_id: (prev_tracker_id, curr_tracker_id)}}}
    """
    history = {}

    for seq_name, acc in accumulators.items():
        seq_history = defaultdict(dict)
        obj_tracker_map = {}  # Maps GT object ID to current tracker ID

        # Process events chronologically
        events_df = acc.events

        for (frame_id, event_id), event in events_df.iterrows():
            event_type = event['Type']
            obj_id = int(event['OId']) if not np.isnan(event['OId']) else None
            tracker_id = int(event['HId']) if not np.isnan(event['HId']) else None

            if event_type in ['MATCH', 'SWITCH'] and obj_id is not None and tracker_id is not None:
                prev_tracker_id = obj_tracker_map.get(obj_id, None)

                if event_type == 'SWITCH':
                    # Record the switch: previous ID -> new ID
                    seq_history[frame_id][obj_id] = (prev_tracker_id, tracker_id)

                # Update current mapping
                obj_tracker_map[obj_id] = tracker_id

        history[seq_name] = seq_history

    return history


def draw_warning_icon(img, x, y, radius=15):
    """
    Draw a warning icon (red circle with white exclamation mark).

    Args:
        img: OpenCV image
        x, y: Center position for the icon
        radius: Radius of the warning circle
    """
    # Draw red filled circle
    cv2.circle(img, (x, y), radius, (0, 0, 255), -1)

    # Draw white exclamation mark
    # Vertical line
    cv2.line(img, (x, y - radius + 5), (x, y + 3), (255, 255, 255), 2)
    # Dot
    cv2.circle(img, (x, y + 8), 2, (255, 255, 255), -1)


def visualize_sequence(seq_name, img_dict, tracking_results, switch_history, output_dir, color_list):
    """
    Visualize one sequence with ID switches highlighted.

    Args:
        seq_name: Sequence name (e.g., 'MOT17-02-FRCNN')
        img_dict: {frame_id: image_path} for this sequence
        tracking_results: {frame_id: [(x1, y1, x2, y2, tracker_id, conf), ...]}
        switch_history: {frame_id: {gt_obj_id: (prev_tracker_id, curr_tracker_id)}}
        output_dir: Directory to save output images
        color_list: Color palette for track IDs
    """
    logger.info(f'Visualizing {seq_name}...')

    # Build reverse map: tracker_id -> gt_obj_id for frames with switches
    # This helps identify which tracker IDs experienced switches
    tracker_to_obj_switches = defaultdict(lambda: defaultdict(dict))

    for frame_id, obj_switches in switch_history.items():
        for obj_id, (prev_tracker_id, curr_tracker_id) in obj_switches.items():
            # Mark the current tracker ID as having a switch
            tracker_to_obj_switches[frame_id][curr_tracker_id] = {
                'gt_obj_id': obj_id,
                'prev_tracker_id': prev_tracker_id,
                'curr_tracker_id': curr_tracker_id
            }

    # Process all frames
    for frame_id in sorted(img_dict.keys()):
        img_path = img_dict[frame_id]

        if not os.path.exists(img_path):
            logger.warning(f'Image not found: {img_path}')
            continue

        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f'Failed to load image: {img_path}')
            continue

        # Get detections for this frame
        detections = tracking_results.get(frame_id, [])

        # Separate normal and switched detections
        switched_detections = []
        normal_detections = []

        for det in detections:
            x1, y1, x2, y2, tracker_id, conf = det

            if tracker_id in tracker_to_obj_switches[frame_id]:
                switched_detections.append(det)
            else:
                normal_detections.append(det)

        # Draw normal detections first (lower z-order)
        for x1, y1, x2, y2, tracker_id, conf in normal_detections:
            color = color_list[tracker_id % 79].tolist()
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            cv2.putText(img, f"{tracker_id}", (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw switched detections on top with highlighting
        for x1, y1, x2, y2, tracker_id, conf in switched_detections:
            switch_info = tracker_to_obj_switches[frame_id][tracker_id]
            gt_obj_id = switch_info['gt_obj_id']
            prev_tracker_id = switch_info['prev_tracker_id']
            curr_tracker_id = switch_info['curr_tracker_id']

            # RED color for switched boxes with thicker border
            switch_color = (0, 0, 255)  # BGR red
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), switch_color, thickness=4)

            # Text annotation: "GT:obj_id  prev→curr"
            text = f"GT:{gt_obj_id} {prev_tracker_id}->{curr_tracker_id}"
            text_y = max(int(y1) - 25, 20)  # Position above box

            # Draw text background for readability
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (int(x1), text_y - text_h - 5),
                         (int(x1) + text_w + 5, text_y + 5),
                         (0, 0, 0), -1)  # Black background

            cv2.putText(img, text, (int(x1), text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw warning icon at top-right of bbox
            icon_x = int(x2) - 20
            icon_y = int(y1) + 20
            draw_warning_icon(img, icon_x, icon_y, radius=12)

        # Save frame
        output_path = os.path.join(output_dir, f"{seq_name}{frame_id:06d}.png")
        cv2.imwrite(output_path, img)

    logger.info(f'{seq_name} visualization complete')


def images_to_video(output_dir, fps=16, size=(1920, 1080)):
    """
    Compile images into video file.

    Args:
        output_dir: Directory containing frame images
        fps: Frames per second
        size: Video resolution (width, height)
    """
    logger.info('Compiling images to video...')

    img_paths = sorted(glob.glob(os.path.join(output_dir, '*.png')))

    if len(img_paths) == 0:
        logger.warning('No images found for video compilation')
        return None

    video_path = output_dir + '_video.avi'
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, size)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, size)
        video_writer.write(img_resized)

    video_writer.release()
    logger.info(f'Video saved to: {video_path}')

    return video_path


def visualize_id_switches(results_folder=None, results_file=None,
                          gt_folder='datasets/mot/train', gt_type='_val_half',
                          json_path=None, output_dir='visual_id_switches',
                          generate_video=True, fps=16, video_size=(1920, 1080)):
    """
    Main visualization function.

    Args:
        results_folder: Path to tracking results folder
        results_file: Path to specific tracking result file
        gt_folder: Path to ground truth folder
        gt_type: Ground truth type suffix
        json_path: Path to COCO JSON annotations (auto-detected if None)
        output_dir: Output directory for visualizations
        generate_video: Whether to compile video
        fps: Video frame rate
        video_size: Video resolution (width, height)
    """
    logger.info('='*80)
    logger.info('Starting ID Switch Visualization')
    logger.info('='*80)

    # Load tracking data
    logger.info('Loading tracking data...')
    gt, ts, gtfiles, tsfiles, results_folder = load_tracking_data(
        results_folder, results_file, gt_folder, gt_type
    )

    if gt is None:
        logger.error('Failed to load tracking data')
        return None

    logger.info(f'Found {len(gtfiles)} ground truth files and {len(tsfiles)} tracking result files')

    # Create accumulators
    logger.info('Creating accumulators...')
    accumulators = create_accumulators(gt, ts)

    # Extract ID switches
    logger.info('Extracting ID switches...')
    switches_by_sequence = extract_id_switches(accumulators)

    # Get tracker history for visualization
    logger.info('Building tracker ID history...')
    tracker_history = get_tracker_history(accumulators)

    # Auto-detect COCO JSON path if not provided
    if json_path is None:
        json_path = f'datasets/mot/annotations/val_half.json' if gt_type == '_val_half' else 'datasets/mot/annotations/test.json'

    if not os.path.exists(json_path):
        logger.error(f'COCO JSON not found: {json_path}')
        return None

    # Load COCO annotations
    logger.info(f'Loading COCO annotations from {json_path}...')
    img_dict = load_coco_annotations(json_path, gt_folder)

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get color palette
    color_list = colormap()

    # Visualize each sequence
    logger.info('\n' + '='*80)
    logger.info('GENERATING VISUALIZATIONS')
    logger.info('='*80)

    total_switches = 0
    for seq_name in ts.keys():
        if seq_name not in img_dict:
            logger.warning(f'No images found for {seq_name}, skipping visualization')
            continue

        # Find corresponding tracking result file
        tracking_file = None
        for tsfile in tsfiles:
            if seq_name in tsfile:
                tracking_file = tsfile
                break

        if tracking_file is None:
            logger.warning(f'No tracking file found for {seq_name}')
            continue

        # Parse tracking results
        tracking_results = parse_tracking_results(tracking_file)

        # Get switch history for this sequence
        switch_history = tracker_history.get(seq_name, {})

        seq_switches = switches_by_sequence[seq_name]['num_switches']
        total_switches += seq_switches

        logger.info(f'\n{seq_name}: {seq_switches} ID switches')

        # Visualize
        visualize_sequence(
            seq_name,
            img_dict[seq_name],
            tracking_results,
            switch_history,
            output_dir,
            color_list
        )

    # Generate video if requested
    video_path = None
    if generate_video:
        logger.info('\n' + '='*80)
        video_path = images_to_video(output_dir, fps, video_size)

    # Summary
    logger.info('\n' + '='*80)
    logger.info('VISUALIZATION SUMMARY')
    logger.info('='*80)
    logger.info(f'Total ID switches visualized: {total_switches}')
    logger.info(f'Frame images saved to: {output_dir}/')
    if video_path:
        logger.info(f'Video saved to: {video_path}')
    logger.info('='*80)

    return {
        'output_dir': output_dir,
        'video_path': video_path,
        'total_switches': total_switches
    }


def make_parser():
    """Create argument parser with detailed help."""
    parser = argparse.ArgumentParser(
        description='Visualize ID switches in MOT tracking results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python3 tools/visualize_id_switches.py

  # Specify custom tracking results folder
  python3 tools/visualize_id_switches.py -r YOLOX_outputs/yolox_x_ablation/track_results

  # Custom output directory
  python3 tools/visualize_id_switches.py -o visual_switches_custom

  # Generate images only (no video)
  python3 tools/visualize_id_switches.py --no-video

  # Custom video settings
  python3 tools/visualize_id_switches.py --fps 30 --size 1280 720

  # Full example with all options
  python3 tools/visualize_id_switches.py \
      -r YOLOX_outputs/yolox_x_ablation/track_results \
      -g datasets/mot/train \
      -t _val_half \
      -o visual_switches \
      --fps 25 \
      --size 1920 1080

Output:
  - Individual frame images: <output_dir>/<sequence><frame_id>.png
  - Video file: <output_dir>_video.avi

Visual Annotations:
  - Normal tracks: Colored boxes with ID labels
  - ID switches: RED thick borders + warning icon + "GT:X prev->curr" text
        """
    )

    parser.add_argument(
        '-r', '--results-folder',
        type=str,
        default='YOLOX_outputs/yolox_x_mot17_half/track_results',
        help='Path to tracking results folder (default: YOLOX_outputs/yolox_x_mot17_half/track_results)'
    )

    parser.add_argument(
        '--results-file',
        type=str,
        default=None,
        help='Path to a specific tracking result file (overrides --results-folder)'
    )

    parser.add_argument(
        '-g', '--gt-folder',
        type=str,
        default='datasets/mot/train',
        help='Path to ground truth folder containing images (default: datasets/mot/train)'
    )

    parser.add_argument(
        '-t', '--gt-type',
        type=str,
        default='_val_half',
        help='Ground truth type suffix: "_val_half", "" (empty), etc. (default: _val_half)'
    )

    parser.add_argument(
        '-j', '--json-path',
        type=str,
        default=None,
        help='Path to COCO JSON annotations (auto-detected from --gt-type if not specified)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='visual_id_switches',
        help='Output directory for visualization frames (default: visual_id_switches)'
    )

    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Skip video generation, only save individual frames'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=16,
        help='Video frame rate (default: 16)'
    )

    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[1920, 1080],
        metavar=('WIDTH', 'HEIGHT'),
        help='Video resolution width and height (default: 1920 1080)'
    )

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    logger.info('Command-line arguments:')
    logger.info(f'  Results folder: {args.results_folder}')
    logger.info(f'  Results file: {args.results_file}')
    logger.info(f'  Ground truth folder: {args.gt_folder}')
    logger.info(f'  Ground truth type: {args.gt_type}')
    logger.info(f'  JSON path: {args.json_path}')
    logger.info(f'  Output directory: {args.output_dir}')
    logger.info(f'  Generate video: {not args.no_video}')
    logger.info(f'  Video FPS: {args.fps}')
    logger.info(f'  Video size: {args.size[0]}x{args.size[1]}')
    logger.info('')

    # Run visualization
    result = visualize_id_switches(
        results_folder=args.results_folder if not args.results_file else None,
        results_file=args.results_file,
        gt_folder=args.gt_folder,
        gt_type=args.gt_type,
        json_path=args.json_path,
        output_dir=args.output_dir,
        generate_video=not args.no_video,
        fps=args.fps,
        video_size=tuple(args.size)
    )

    if result:
        logger.info('\n✓ Visualization completed successfully!')
    else:
        logger.error('\n✗ Visualization failed')
        sys.exit(1)
