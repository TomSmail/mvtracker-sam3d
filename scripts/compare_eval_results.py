#!/usr/bin/env python3
"""
Compare baseline vs SAM3D evaluation results.

Usage:
    python scripts/compare_eval_results.py \\
        --baseline logs/mvtracker_human_eval_baseline \\
        --sam3d logs/mvtracker_human_eval_sam3d \\
        --output comparison_results.csv
"""

import argparse
import glob
import pandas as pd
from pathlib import Path


def load_metrics(log_dir):
    """Load all metrics CSVs from evaluation log directory."""
    csv_files = glob.glob(f"{log_dir}/eval_*/step-*_metrics_avg.csv")

    if not csv_files:
        print(f"Warning: No metrics found in {log_dir}")
        return []

    results = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        # Extract dataset name from path
        dataset_name = Path(csv_file).parent.name.replace("eval_", "")
        df['dataset'] = dataset_name
        df['log_dir'] = log_dir
        results.append(df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compare_results(baseline_df, sam3d_df, key_metrics=None):
    """Compare baseline vs SAM3D metrics."""
    if key_metrics is None:
        key_metrics = [
            'average_jaccard',
            'occlusion_accuracy',
            'survival',
            'average_pts_within_thresh',
            '<deltax>',
        ]

    # Filter to only the key metrics that exist
    existing_metrics = [m for m in key_metrics if m in baseline_df.columns]

    comparison = []

    for dataset in baseline_df['dataset'].unique():
        baseline_row = baseline_df[baseline_df['dataset'] == dataset]
        sam3d_row = sam3d_df[sam3d_df['dataset'] == dataset]

        if baseline_row.empty or sam3d_row.empty:
            print(f"Warning: Missing data for dataset {dataset}")
            continue

        result = {'dataset': dataset}

        for metric in existing_metrics:
            baseline_val = baseline_row[metric].values[0]
            sam3d_val = sam3d_row[metric].values[0]
            improvement = sam3d_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0

            result[f'{metric}_baseline'] = baseline_val
            result[f'{metric}_sam3d'] = sam3d_val
            result[f'{metric}_improvement'] = improvement
            result[f'{metric}_improvement_pct'] = improvement_pct

        comparison.append(result)

    return pd.DataFrame(comparison)


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs SAM3D evaluation results')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline log directory')
    parser.add_argument('--sam3d', type=str, required=True,
                        help='Path to SAM3D log directory')
    parser.add_argument('--output', type=str, default='comparison_results.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    print(f"Loading baseline results from: {args.baseline}")
    baseline_df = load_metrics(args.baseline)

    print(f"Loading SAM3D results from: {args.sam3d}")
    sam3d_df = load_metrics(args.sam3d)

    if baseline_df.empty or sam3d_df.empty:
        print("Error: No metrics found in one or both directories")
        return

    print("\n=== Baseline Results ===")
    print(baseline_df[['dataset', 'average_jaccard', 'occlusion_accuracy', 'survival']].to_string())

    print("\n=== SAM3D Results ===")
    print(sam3d_df[['dataset', 'average_jaccard', 'occlusion_accuracy', 'survival']].to_string())

    print("\n=== Comparing Results ===")
    comparison_df = compare_results(baseline_df, sam3d_df)

    if not comparison_df.empty:
        print("\n=== Summary ===")
        print(comparison_df.to_string())

        # Save to CSV
        comparison_df.to_csv(args.output, index=False)
        print(f"\nFull comparison saved to: {args.output}")

        # Print key findings
        print("\n=== Key Improvements (%) ===")
        for metric in ['average_jaccard', 'occlusion_accuracy', 'survival']:
            improvement_col = f'{metric}_improvement_pct'
            if improvement_col in comparison_df.columns:
                avg_improvement = comparison_df[improvement_col].mean()
                print(f"{metric:30s}: {avg_improvement:+.2f}%")
    else:
        print("Error: Could not generate comparison")


if __name__ == '__main__':
    main()
