#!/usr/bin/env python3
"""
Analyze ID Switches in MOT Tracking Results

This script analyzes tracking results to identify:
1. Which function calculates the IDs (num_switches) field
2. Which frames have ID switches occurring

Usage Examples:
    # Basic usage with defaults
    python3 tools/analyze_id_switches.py

    # Specify custom tracking results folder
    python3 tools/analyze_id_switches.py -r YOLOX_outputs/yolox_x_ablation/track_results

    # Specify custom ground truth folder
    python3 tools/analyze_id_switches.py -g datasets/mot/train

    # Specify ground truth type (empty for test set, '_val_half' for validation)
    python3 tools/analyze_id_switches.py -t ''

    # Analyze specific tracking result file
    python3 tools/analyze_id_switches.py --results-file YOLOX_outputs/yolox_x_ablation/track_results/MOT17-02-FRCNN.txt

    # Combine options
    python3 tools/analyze_id_switches.py -r YOLOX_outputs/bytetrack_sort/track_results_sort -g datasets/mot/train -t '_val_half'
"""

from loguru import logger
import motmetrics as mm
import pandas as pd
import os
import glob
import argparse
from collections import OrderedDict
from pathlib import Path


def analyze_id_switches(results_folder=None, results_file=None, gt_folder='datasets/mot/train', gt_type='_val_half'):
    """
    Analyze ID switches in tracking results.

    Args:
        results_folder: Path to tracking results folder (default: YOLOX_outputs/yolox_x_mot17_half/track_results)
        results_file: Path to a specific tracking result file (overrides results_folder)
        gt_folder: Path to ground truth folder (default: datasets/mot/train)
        gt_type: Ground truth type suffix (e.g., '_val_half', '') (default: '_val_half')
    """

    # Setup
    mm.lap.default_solver = 'lap'

    # Load ground truth files
    gtfiles = glob.glob(
        os.path.join(gt_folder, '*/gt/gt{}.txt'.format(gt_type)))

    # Load tracking result files
    if results_file:
        # Single file mode
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            return
        tsfiles = [results_file]
        results_folder = os.path.dirname(results_file)
    elif results_folder:
        # Folder mode
        if not os.path.exists(results_folder):
            logger.error(f"Results folder not found: {results_folder}")
            return
        tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt'))
                   if not os.path.basename(f).startswith('eval')]
    else:
        logger.error("Either results_folder or results_file must be specified")
        return

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))

    if len(gtfiles) == 0 or len(tsfiles) == 0:
        logger.error("No ground truth or tracking files found!")
        return

    # Load data into motmetrics format
    logger.info('Loading ground truth and tracking results...')
    gt = OrderedDict([
        (Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1))
        for f in gtfiles
    ])
    ts = OrderedDict([
        (os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1.0))
        for f in tsfiles
    ])

    # Create accumulators for each sequence
    logger.info('Creating accumulators and computing metrics...')
    accumulators = OrderedDict()

    for seq_name in ts.keys():
        if seq_name in gt:
            logger.info(f'Processing {seq_name}...')
            # Create accumulator by comparing to ground truth
            acc = mm.utils.compare_to_groundtruth(gt[seq_name], ts[seq_name], 'iou', distth=0.5)
            accumulators[seq_name] = acc
        else:
            logger.warning(f'No ground truth for {seq_name}, skipping.')

    # Analyze ID switches for each sequence
    logger.info('\n' + '='*80)
    logger.info('ID SWITCH ANALYSIS RESULTS')
    logger.info('='*80)

    all_switches_data = []

    for seq_name, acc in accumulators.items():
        logger.info(f'\n{seq_name}:')
        logger.info('-' * 60)

        # Get all events
        events_df = acc.events

        # Filter for SWITCH events only
        switches_df = events_df[events_df['Type'] == 'SWITCH']

        # Total number of switches
        num_switches = len(switches_df)
        logger.info(f'Total ID switches: {num_switches}')

        if num_switches == 0:
            logger.info('No ID switches detected in this sequence.')
            continue

        # Get frames with switches
        frames_with_switches = switches_df.index.get_level_values(0).unique().tolist()
        logger.info(f'Number of frames with switches: {len(frames_with_switches)}')
        logger.info(f'Frame IDs with switches: {frames_with_switches[:20]}{"..." if len(frames_with_switches) > 20 else ""}')

        # Detailed analysis of switches
        logger.info('\nDetailed switch information (showing first 10):')

        switch_count = 0
        for frame_id in sorted(frames_with_switches)[:10]:
            frame_switches = switches_df.loc[frame_id]

            # Handle both single row and multiple rows
            if isinstance(frame_switches, pd.Series):
                frame_switches = frame_switches.to_frame().T

            for idx, switch in frame_switches.iterrows():
                obj_id = int(switch['OId']) if not pd.isna(switch['OId']) else 'Unknown'
                hyp_id = int(switch['HId']) if not pd.isna(switch['HId']) else 'Unknown'
                iou_dist = switch['D'] if not pd.isna(switch['D']) else 'N/A'

                logger.info(f'  Frame {frame_id}: Object {obj_id} → Tracker {hyp_id} (IoU distance: {iou_dist:.3f})')

                # Collect data for CSV export
                all_switches_data.append({
                    'Sequence': seq_name,
                    'Frame': frame_id,
                    'Object_ID': obj_id,
                    'Tracker_ID': hyp_id,
                    'IoU_Distance': iou_dist
                })

                switch_count += 1

        if num_switches > 10:
            logger.info(f'  ... and {num_switches - switch_count} more switches')

    # Export to CSV
    if all_switches_data:
        csv_path = os.path.join(results_folder, 'id_switches_analysis.csv')
        switches_export_df = pd.DataFrame(all_switches_data)
        switches_export_df.to_csv(csv_path, index=False)
        logger.info(f'\n✓ Detailed switch data exported to: {csv_path}')

    # Summary statistics
    logger.info('\n' + '='*80)
    logger.info('SUMMARY')
    logger.info('='*80)

    # Compute overall metrics
    mh = mm.metrics.create()
    metrics = ['num_switches', 'num_objects']
    summary = mh.compute_many(
        list(accumulators.values()),
        names=list(accumulators.keys()),
        metrics=metrics,
        generate_overall=True
    )

    logger.info('\nID Switches per sequence:')
    for seq_name in accumulators.keys():
        switches = summary.loc[seq_name, 'num_switches']
        objects = summary.loc[seq_name, 'num_objects']
        logger.info(f'  {seq_name}: {int(switches)} switches out of {int(objects)} total detections')

    logger.info(f'\nOVERALL: {int(summary.loc["OVERALL", "num_switches"])} total ID switches')

    # Answer the two key questions
    logger.info('\n' + '='*80)
    logger.info('ANSWERS TO KEY QUESTIONS')
    logger.info('='*80)
    logger.info('\n1. Which function calculates the IDs field?')
    logger.info('   → motmetrics.metrics.num_switches()')
    logger.info('   → Located in: motmetrics/metrics.py')
    logger.info('   → Counts events where Type == "SWITCH" in the accumulator events DataFrame')

    logger.info('\n2. Which frames have ID switches?')
    logger.info('   → See detailed output above for frame-by-frame breakdown')
    logger.info(f'   → Full details exported to: {csv_path if all_switches_data else "N/A"}')

    logger.info('\n' + '='*80)


def make_parser():
    """Create argument parser with detailed help."""
    parser = argparse.ArgumentParser(
        description='Analyze ID switches in MOT tracking results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python3 tools/analyze_id_switches.py

  # Specify custom tracking results folder
  python3 tools/analyze_id_switches.py -r YOLOX_outputs/yolox_x_ablation/track_results

  # Analyze ByteTrack results
  python3 tools/analyze_id_switches.py -r YOLOX_outputs/yolox_x_ablation/track_results -g datasets/mot/train -t _val_half

  # Analyze SORT tracker results
  python3 tools/analyze_id_switches.py -r YOLOX_outputs/yolox_x_ablation/track_results_sort

  # Analyze DeepSORT tracker results
  python3 tools/analyze_id_switches.py -r YOLOX_outputs/yolox_x_ablation/track_results_deepsort

  # Analyze specific tracking result file
  python3 tools/analyze_id_switches.py --results-file YOLOX_outputs/yolox_x_ablation/track_results/MOT17-02-FRCNN.txt

  # Use test set ground truth (empty suffix)
  python3 tools/analyze_id_switches.py -t ''

  # Custom ground truth location
  python3 tools/analyze_id_switches.py -g datasets/MOT20/train -t _val_half

Output:
  - Console output with detailed switch information
  - CSV file: <results_folder>/id_switches_analysis.csv
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
        help='Path to ground truth folder containing subdirectories with gt/gt*.txt files (default: datasets/mot/train)'
    )

    parser.add_argument(
        '-t', '--gt-type',
        type=str,
        default='_val_half',
        help='Ground truth file suffix: "_val_half" for validation set, "" (empty) for test set, "_train_half" for train half (default: _val_half)'
    )

    return parser


if __name__ == '__main__':
    # Parse command-line arguments
    parser = make_parser()
    args = parser.parse_args()

    logger.info('='*80)
    logger.info('Starting ID Switch Analysis')
    logger.info('='*80)

    if args.results_file:
        logger.info(f'Results file: {args.results_file}')
    else:
        logger.info(f'Results folder: {args.results_folder}')

    logger.info(f'Ground truth folder: {args.gt_folder}')
    logger.info(f'Ground truth type: {args.gt_type}')
    logger.info('')

    # Run analysis
    analyze_id_switches(
        results_folder=args.results_folder if not args.results_file else None,
        results_file=args.results_file,
        gt_folder=args.gt_folder,
        gt_type=args.gt_type
    )

    logger.info('\n' + '='*80)
    logger.info('Analysis completed!')
    logger.info('='*80)
