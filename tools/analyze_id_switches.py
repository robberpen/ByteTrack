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


def load_tracking_data(results_folder=None, results_file=None, gt_folder='datasets/mot/train', gt_type='_val_half'):
    """
    Load ground truth and tracking result files.

    Args:
        results_folder: Path to tracking results folder
        results_file: Path to a specific tracking result file (overrides results_folder)
        gt_folder: Path to ground truth folder
        gt_type: Ground truth type suffix (e.g., '_val_half', '')

    Returns:
        tuple: (gt_dict, ts_dict, gtfiles, tsfiles, results_folder)
               Returns (None, None, None, None, None) on error
    """
    mm.lap.default_solver = 'lap'

    # Load ground truth files
    gtfiles = glob.glob(os.path.join(gt_folder, '*/gt/gt{}.txt'.format(gt_type)))

    # Load tracking result files
    if results_file:
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            return None, None, None, None, None
        tsfiles = [results_file]
        results_folder = os.path.dirname(results_file)
    elif results_folder:
        if not os.path.exists(results_folder):
            logger.error(f"Results folder not found: {results_folder}")
            return None, None, None, None, None
        tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt'))
                   if not os.path.basename(f).startswith('eval')]
    else:
        logger.error("Either results_folder or results_file must be specified")
        return None, None, None, None, None

    if len(gtfiles) == 0 or len(tsfiles) == 0:
        logger.error("No ground truth or tracking files found!")
        return None, None, None, None, None

    # Load data into motmetrics format
    gt = OrderedDict([
        (Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1))
        for f in gtfiles
    ])
    ts = OrderedDict([
        (os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1.0))
        for f in tsfiles
    ])

    return gt, ts, gtfiles, tsfiles, results_folder


def create_accumulators(gt, ts):
    """
    Create motmetrics accumulators by comparing tracking results to ground truth.

    Args:
        gt: OrderedDict of ground truth data (seq_name -> DataFrame)
        ts: OrderedDict of tracking result data (seq_name -> DataFrame)

    Returns:
        OrderedDict: accumulators (seq_name -> MOTAccumulator)
    """
    accumulators = OrderedDict()

    for seq_name in ts.keys():
        if seq_name in gt:
            acc = mm.utils.compare_to_groundtruth(gt[seq_name], ts[seq_name], 'iou', distth=0.5)
            accumulators[seq_name] = acc
        else:
            logger.warning(f'No ground truth for {seq_name}, skipping.')

    return accumulators


def extract_id_switches(accumulators):
    """
    Extract ID switch events from accumulators.

    Args:
        accumulators: OrderedDict of MOTAccumulators (seq_name -> accumulator)

    Returns:
        dict: Nested dict structure:
              {
                  seq_name: {
                      'switches_df': DataFrame of SWITCH events,
                      'frames_with_switches': list of frame IDs,
                      'num_switches': int,
                      'switches_data': list of dicts with detailed switch info
                  }
              }
    """
    switches_by_sequence = {}

    for seq_name, acc in accumulators.items():
        events_df = acc.events
        switches_df = events_df[events_df['Type'] == 'SWITCH']
        num_switches = len(switches_df)

        if num_switches == 0:
            switches_by_sequence[seq_name] = {
                'switches_df': switches_df,
                'frames_with_switches': [],
                'num_switches': 0,
                'switches_data': []
            }
            continue

        frames_with_switches = switches_df.index.get_level_values(0).unique().tolist()

        # Extract detailed switch information
        switches_data = []
        for frame_id in frames_with_switches:
            frame_switches = switches_df.loc[frame_id]

            # Handle both single row and multiple rows
            if isinstance(frame_switches, pd.Series):
                frame_switches = frame_switches.to_frame().T

            for idx, switch in frame_switches.iterrows():
                obj_id = int(switch['OId']) if not pd.isna(switch['OId']) else None
                hyp_id = int(switch['HId']) if not pd.isna(switch['HId']) else None
                iou_dist = switch['D'] if not pd.isna(switch['D']) else None

                switches_data.append({
                    'frame': frame_id,
                    'object_id': obj_id,
                    'tracker_id': hyp_id,
                    'iou_distance': iou_dist
                })

        switches_by_sequence[seq_name] = {
            'switches_df': switches_df,
            'frames_with_switches': frames_with_switches,
            'num_switches': num_switches,
            'switches_data': switches_data
        }

    return switches_by_sequence


def export_switches_csv(switches_by_sequence, output_path):
    """
    Export ID switches to CSV file.

    Args:
        switches_by_sequence: dict returned by extract_id_switches()
        output_path: Path to save CSV file

    Returns:
        str: Path to saved CSV file, or None if no data
    """
    all_switches_data = []

    for seq_name, seq_data in switches_by_sequence.items():
        for switch in seq_data['switches_data']:
            all_switches_data.append({
                'Sequence': seq_name,
                'Frame': switch['frame'],
                'Object_ID': switch['object_id'],
                'Tracker_ID': switch['tracker_id'],
                'IoU_Distance': switch['iou_distance']
            })

    if all_switches_data:
        switches_export_df = pd.DataFrame(all_switches_data)
        switches_export_df.to_csv(output_path, index=False)
        return output_path

    return None


def analyze_id_switches(results_folder=None, results_file=None, gt_folder='datasets/mot/train', gt_type='_val_half'):
    """
    Analyze ID switches in tracking results.

    Args:
        results_folder: Path to tracking results folder (default: YOLOX_outputs/yolox_x_mot17_half/track_results)
        results_file: Path to a specific tracking result file (overrides results_folder)
        gt_folder: Path to ground truth folder (default: datasets/mot/train)
        gt_type: Ground truth type suffix (e.g., '_val_half', '') (default: '_val_half')

    Returns:
        dict: Analysis results containing:
              - 'gt': Ground truth data
              - 'ts': Tracking result data
              - 'accumulators': MOTAccumulators
              - 'switches': ID switch data by sequence
              - 'summary': Overall metrics summary
              - 'csv_path': Path to exported CSV (if any)
              Returns None on error
    """
    # Load data
    gt, ts, gtfiles, tsfiles, results_folder = load_tracking_data(results_folder, results_file, gt_folder, gt_type)
    if gt is None:
        return None

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Loading ground truth and tracking results...')

    # Create accumulators
    logger.info('Creating accumulators and computing metrics...')
    accumulators = create_accumulators(gt, ts)

    # Extract ID switches
    switches_by_sequence = extract_id_switches(accumulators)

    # Log analysis results
    logger.info('\n' + '='*80)
    logger.info('ID SWITCH ANALYSIS RESULTS')
    logger.info('='*80)

    for seq_name, seq_data in switches_by_sequence.items():
        logger.info(f'\n{seq_name}:')
        logger.info('-' * 60)
        logger.info(f'Total ID switches: {seq_data["num_switches"]}')

        if seq_data['num_switches'] == 0:
            logger.info('No ID switches detected in this sequence.')
            continue

        frames = seq_data['frames_with_switches']
        logger.info(f'Number of frames with switches: {len(frames)}')
        logger.info(f'Frame IDs with switches: {frames[:20]}{"..." if len(frames) > 20 else ""}')

        logger.info('\nDetailed switch information (showing first 10):')
        for switch in seq_data['switches_data'][:10]:
            logger.info(f'  Frame {switch["frame"]}: Object {switch["object_id"]} → '
                       f'Tracker {switch["tracker_id"]} (IoU distance: {switch["iou_distance"]:.3f})')

        if seq_data['num_switches'] > 10:
            logger.info(f'  ... and {seq_data["num_switches"] - 10} more switches')

    # Export to CSV
    csv_path = None
    if any(seq_data['num_switches'] > 0 for seq_data in switches_by_sequence.values()):
        csv_path = os.path.join(results_folder, 'id_switches_analysis.csv')
        export_switches_csv(switches_by_sequence, csv_path)
        logger.info(f'\n✓ Detailed switch data exported to: {csv_path}')

    # Summary statistics
    logger.info('\n' + '='*80)
    logger.info('SUMMARY')
    logger.info('='*80)

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
    logger.info(f'   → Full details exported to: {csv_path if csv_path else "N/A"}')

    logger.info('\n' + '='*80)

    # Return analysis results for programmatic use
    return {
        'gt': gt,
        'ts': ts,
        'accumulators': accumulators,
        'switches': switches_by_sequence,
        'summary': summary,
        'csv_path': csv_path
    }


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
