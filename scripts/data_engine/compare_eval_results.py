"""
Compare evaluation results between baseline MVTracker and SAM3D-guided MVTracker.

Reads metrics CSV files from two experiment directories and computes per-dataset
deltas. Expects the directory structure produced by train.py evaluation:
  {experiment_path}/eval_{dataset_name}/step-{step}_metrics_avg.csv

Usage:
    python scripts/data_engine/compare_eval_results.py \
        --baseline_dir logs/mvtracker_human_eval_baseline \
        --sam3d_dir logs/mvtracker_human_eval_sam3d

    # Compare a specific step from finetuning against baseline:
    python scripts/data_engine/compare_eval_results.py \
        --baseline_dir logs/mvtracker_human_eval_baseline \
        --sam3d_dir logs/mvtracker_finetune_sam3d \
        --sam3d_step 5000
"""

import argparse
import os
import re
import sys

import pandas as pd


def find_latest_metrics(eval_dir, step=None):
    """
    Find the metrics_avg.csv with the highest step number in eval_dir,
    or a specific step if provided.

    Returns:
        (filepath, step_number) or (None, -1) if not found
    """
    if not os.path.isdir(eval_dir):
        return None, -1
    pattern = re.compile(r"step-(\d+)_metrics_avg\.csv")
    best_file = None
    best_step = -1
    for fname in os.listdir(eval_dir):
        m = pattern.match(fname)
        if not m:
            continue
        s = int(m.group(1))
        if step is not None:
            if s == step:
                return os.path.join(eval_dir, fname), s
        elif s > best_step:
            best_step = s
            best_file = os.path.join(eval_dir, fname)
    if step is not None:
        return None, -1
    return best_file, best_step


def load_metrics(experiment_dir, dataset_name, step=None):
    """Load metrics for a dataset from an experiment directory."""
    eval_dir = os.path.join(experiment_dir, f"eval_{dataset_name}")
    filepath, found_step = find_latest_metrics(eval_dir, step=step)
    if filepath is None:
        return None, -1
    df = pd.read_csv(filepath, index_col=0)
    metrics = df["score"].to_dict()
    return metrics, found_step


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs SAM3D evaluation results")
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Experiment directory for baseline evaluation")
    parser.add_argument("--sam3d_dir", type=str, required=True,
                        help="Experiment directory for SAM3D evaluation")
    parser.add_argument("--baseline_step", type=int, default=None,
                        help="Specific step to use for baseline (default: latest)")
    parser.add_argument("--sam3d_step", type=int, default=None,
                        help="Specific step to use for SAM3D (default: latest)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Dataset names to compare (default: auto-detect)")
    args = parser.parse_args()

    # Auto-detect datasets from baseline directory
    if args.datasets is None:
        datasets = []
        for entry in sorted(os.listdir(args.baseline_dir)):
            if entry.startswith("eval_") and os.path.isdir(os.path.join(args.baseline_dir, entry)):
                datasets.append(entry.replace("eval_", "", 1))
        if not datasets:
            print(f"No eval_* directories found in {args.baseline_dir}")
            sys.exit(1)
    else:
        datasets = args.datasets

    print(f"Baseline: {args.baseline_dir}")
    print(f"SAM3D:    {args.sam3d_dir}")
    print(f"Datasets: {datasets}")
    print()

    # Key metrics to highlight (higher is better for these)
    higher_is_better = {"3dpt/aj", "3dpt/occ_acc", "3dpt/survival_rate"}
    # Lower is better for these
    lower_is_better = {"3dpt/ate", "3dpt/mte", "3dpt/fde"}

    all_results = []
    for ds in datasets:
        baseline_metrics, baseline_step = load_metrics(args.baseline_dir, ds, step=args.baseline_step)
        sam3d_metrics, sam3d_step = load_metrics(args.sam3d_dir, ds, step=args.sam3d_step)

        if baseline_metrics is None:
            print(f"  {ds}: baseline metrics not found, skipping")
            continue
        if sam3d_metrics is None:
            print(f"  {ds}: SAM3D metrics not found, skipping")
            continue

        print(f"=== {ds} ===")
        print(f"  Baseline step: {baseline_step}, SAM3D step: {sam3d_step}")

        common_keys = sorted(set(baseline_metrics.keys()) & set(sam3d_metrics.keys()))
        for key in common_keys:
            bv = baseline_metrics[key]
            sv = sam3d_metrics[key]
            delta = sv - bv

            if key in higher_is_better:
                indicator = "+" if delta > 0 else "-" if delta < 0 else "="
            elif key in lower_is_better:
                indicator = "+" if delta < 0 else "-" if delta > 0 else "="
            else:
                indicator = ""

            print(f"  {key:<35s}  baseline={bv:>8.4f}  sam3d={sv:>8.4f}  delta={delta:>+8.4f}  {indicator}")

            all_results.append({
                "dataset": ds,
                "metric": key,
                "baseline": bv,
                "sam3d": sv,
                "delta": delta,
            })

        print()

    if not all_results:
        print("No results to summarize.")
        return

    # Summary across all datasets
    df = pd.DataFrame(all_results)
    print("=== Summary (averaged across datasets) ===")
    summary = df.groupby("metric")[["baseline", "sam3d", "delta"]].mean()
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(summary)


if __name__ == "__main__":
    main()
