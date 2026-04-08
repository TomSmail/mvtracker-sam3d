"""
Report data volume and quality metrics for generated human tracks.

This script scans Panoptic sequence folders and summarizes:
- where generated assets live
- how much disk space they occupy
- track quality metrics (visibility, continuity, motion stability)
- potential issues and suggested follow-up actions

Usage:
    python scripts/data_engine/report_human_tracks.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview

    # Single sequence
    python scripts/data_engine/report_human_tracks.py \
        --data_root ... --seq basketball

    # Save machine-readable outputs
    python scripts/data_engine/report_human_tracks.py \
        --data_root ... --save_json reports/human_tracks_report.json \
        --save_markdown reports/human_tracks_report.md
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np


def list_sequences(data_root):
    seqs = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and not d.startswith(".")
        and not d.startswith("_")
    ]
    return sorted(seqs)


def bytes_to_human(n_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def dir_size_bytes(path):
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                pass
    return total


def summarize_sam3d_predictions(seq_path):
    pred_dir = os.path.join(seq_path, "sam3d_predictions")
    if not os.path.isdir(pred_dir):
        return {
            "exists": False,
            "path": pred_dir,
            "n_views_with_predictions": 0,
            "n_prediction_files": 0,
            "size_bytes": 0,
        }

    n_prediction_files = 0
    n_views_with_predictions = 0
    for name in sorted(os.listdir(pred_dir)):
        view_dir = os.path.join(pred_dir, name)
        if not os.path.isdir(view_dir):
            continue
        files = [f for f in os.listdir(view_dir) if f.endswith(".npz")]
        if files:
            n_views_with_predictions += 1
            n_prediction_files += len(files)

    return {
        "exists": True,
        "path": pred_dir,
        "n_views_with_predictions": int(n_views_with_predictions),
        "n_prediction_files": int(n_prediction_files),
        "size_bytes": int(dir_size_bytes(pred_dir)),
    }


def summarize_tracks_file(seq_path, jump_threshold):
    tracks_path = os.path.join(seq_path, "human_tracks.npz")
    if not os.path.exists(tracks_path):
        return {
            "exists": False,
            "path": tracks_path,
            "size_bytes": 0,
        }

    out = {
        "exists": True,
        "path": tracks_path,
        "size_bytes": int(os.path.getsize(tracks_path)),
        "modified_at": datetime.fromtimestamp(os.path.getmtime(tracks_path)).isoformat(),
    }

    with np.load(tracks_path) as data:
        required = ["trajectories", "per_view_visibilities", "track_valid"]
        missing = [k for k in required if k not in data]
        if missing:
            out["error"] = f"Missing keys: {missing}"
            return out

        traj3d = data["trajectories"]                  # [T, N, 3]
        vis = data["per_view_visibilities"]            # [V, T, N]
        valid = data["track_valid"]                    # [T, N]
        traj2d = data["trajectories_pixelspace"] if "trajectories_pixelspace" in data else None

        T, N, _ = traj3d.shape
        V = vis.shape[0]
        n_persons = int(data["n_persons"]) if "n_persons" in data else 0
        n_keypoints = int(data["n_keypoints"]) if "n_keypoints" in data else 70
        n_sampled_verts = int(data["n_sampled_verts"]) if "n_sampled_verts" in data else 0

        visible = vis & valid[None, :, :]
        visible_counts_view_frame = visible.sum(axis=2)  # [V, T]
        visible_counts_frame = visible_counts_view_frame.mean(axis=0)  # [T], averaged over views

        out["n_frames"] = int(T)
        out["n_tracks"] = int(N)
        out["n_views"] = int(V)
        out["n_persons"] = int(n_persons)
        out["n_keypoints"] = int(n_keypoints)
        out["n_sampled_verts"] = int(n_sampled_verts)

        out["valid_ratio"] = float(valid.mean())
        out["visibility_mean"] = float(visible.mean())
        out["visibility_per_view_mean"] = [float(x) for x in visible.mean(axis=(1, 2))]
        out["visible_tracks_per_frame_mean"] = float(visible_counts_frame.mean())
        out["visible_tracks_per_frame_min"] = float(visible_counts_frame.min())
        out["visible_tracks_per_frame_p05"] = float(np.percentile(visible_counts_frame, 5))
        out["visible_tracks_per_frame_p95"] = float(np.percentile(visible_counts_frame, 95))
        out["visible_tracks_per_frame_max"] = float(visible_counts_frame.max())

        out["zero_visible_frames_any_view"] = int((visible_counts_view_frame == 0).sum())
        out["zero_visible_frames_all_views"] = int((visible.sum(axis=(0, 2)) == 0).sum())

        if traj2d is not None and "img_h" in data and "img_w" in data:
            img_h = int(data["img_h"])
            img_w = int(data["img_w"])
            in_bounds = (
                (traj2d[..., 0] >= 0) & (traj2d[..., 0] < img_w)
                & (traj2d[..., 1] >= 0) & (traj2d[..., 1] < img_h)
            )
            out["projected_in_bounds_ratio"] = float(in_bounds.mean())

        # Motion stability from pelvis trajectories
        motion = {}
        if n_persons > 0 and n_keypoints >= 11:
            n_tracks_per_person = n_keypoints + n_sampled_verts
            pelvis_series = []
            for p in range(n_persons):
                base = p * n_tracks_per_person
                if base + 10 >= N:
                    break
                pelvis = traj3d[:, base + 9:base + 11, :].mean(axis=1)  # [T, 3]
                pelvis_series.append(pelvis)
            if pelvis_series:
                pelvis_arr = np.stack(pelvis_series, axis=1)  # [T, P, 3]
                speeds = np.linalg.norm(np.diff(pelvis_arr, axis=0), axis=2)  # [T-1, P]
                if speeds.size > 0:
                    motion["pelvis_speed_mean"] = float(speeds.mean())
                    motion["pelvis_speed_p95"] = float(np.percentile(speeds, 95))
                    motion["pelvis_jump_threshold"] = float(jump_threshold)
                    motion["pelvis_jump_frames"] = int((speeds > jump_threshold).sum())
                    motion["pelvis_jump_ratio"] = float((speeds > jump_threshold).mean())
        out["motion"] = motion

        # Simple identity ambiguity heuristic for >=2 people
        if n_persons >= 2 and n_keypoints >= 11:
            n_tracks_per_person = n_keypoints + n_sampled_verts
            pelvis_arr = np.zeros((T, n_persons, 3), dtype=np.float32)
            valid_people = True
            for p in range(n_persons):
                base = p * n_tracks_per_person
                if base + 10 >= N:
                    valid_people = False
                    break
                pelvis_arr[:, p, :] = traj3d[:, base + 9:base + 11, :].mean(axis=1)

            if valid_people and T > 1:
                ambiguity_count = 0
                for t in range(1, T):
                    prev = pelvis_arr[t - 1]   # [P, 3]
                    curr = pelvis_arr[t]       # [P, 3]
                    cost = np.linalg.norm(prev[:, None, :] - curr[None, :, :], axis=-1)
                    # If matching to another identity is cheaper than staying on diagonal
                    # for any person, this frame is identity-ambiguous.
                    diag = np.diag(cost)
                    offdiag = cost.copy()
                    np.fill_diagonal(offdiag, np.inf)
                    if (offdiag.min(axis=1) < diag).any():
                        ambiguity_count += 1
                out["identity_ambiguous_frames"] = int(ambiguity_count)
                out["identity_ambiguous_ratio"] = float(ambiguity_count / max(T - 1, 1))

    return out


def sequence_report(seq_name, seq_path, jump_threshold):
    sam3d = summarize_sam3d_predictions(seq_path)
    tracks = summarize_tracks_file(seq_path, jump_threshold=jump_threshold)

    issues = []
    recommendations = []

    if not sam3d["exists"]:
        issues.append("sam3d_predictions directory missing")
        recommendations.append("Run run_sam3d_inference.py for this sequence.")
    if not tracks["exists"]:
        issues.append("human_tracks.npz missing")
        recommendations.append("Run generate_human_tracks.py for this sequence.")

    if tracks.get("exists"):
        vis_mean = tracks.get("visibility_mean", 0.0)
        valid_ratio = tracks.get("valid_ratio", 0.0)
        zero_all = tracks.get("zero_visible_frames_all_views", 0)
        jump_ratio = tracks.get("motion", {}).get("pelvis_jump_ratio", 0.0)
        ambiguous_ratio = tracks.get("identity_ambiguous_ratio", 0.0)

        if vis_mean < 0.45:
            issues.append(f"low visibility_mean={vis_mean:.3f}")
            recommendations.append(
                "Check depth maps / occlusion settings and review detector quality in hard views."
            )
        if valid_ratio < 0.99:
            issues.append(f"track_valid ratio below 1.0 ({valid_ratio:.3f})")
            recommendations.append(
                "Inspect low-view-count frames and interpolation settings."
            )
        if zero_all > 0:
            issues.append(f"{zero_all} frames have zero visible tracks in all views")
            recommendations.append(
                "Inspect world-to-camera transforms and outlier rejection."
            )
        if jump_ratio > 0.05:
            issues.append(f"high pelvis jump ratio ({jump_ratio:.3f})")
            recommendations.append(
                "Tighten cross-view matching, increase min views, or adjust smoothing kernel."
            )
        if ambiguous_ratio > 0.1:
            issues.append(f"identity ambiguity ratio is high ({ambiguous_ratio:.3f})")
            recommendations.append(
                "Use stronger temporal identity linking features (appearance/velocity) in addition to pelvis distance."
            )

    return {
        "sequence": seq_name,
        "path": seq_path,
        "sam3d_predictions": sam3d,
        "tracks": tracks,
        "issues": issues,
        "recommendations": recommendations,
    }


def aggregate_report(seq_reports):
    total_pred_bytes = sum(r["sam3d_predictions"]["size_bytes"] for r in seq_reports)
    total_tracks_bytes = sum(r["tracks"]["size_bytes"] for r in seq_reports)
    n_seq = len(seq_reports)
    n_with_tracks = sum(1 for r in seq_reports if r["tracks"]["exists"])
    n_with_preds = sum(1 for r in seq_reports if r["sam3d_predictions"]["exists"])

    vis_values = [
        r["tracks"].get("visibility_mean")
        for r in seq_reports
        if r["tracks"].get("exists") and "visibility_mean" in r["tracks"]
    ]

    return {
        "n_sequences_scanned": n_seq,
        "n_sequences_with_sam3d_predictions": n_with_preds,
        "n_sequences_with_human_tracks": n_with_tracks,
        "total_sam3d_predictions_size_bytes": int(total_pred_bytes),
        "total_human_tracks_size_bytes": int(total_tracks_bytes),
        "total_generated_size_bytes": int(total_pred_bytes + total_tracks_bytes),
        "mean_visibility_across_sequences": float(np.mean(vis_values)) if vis_values else None,
    }


def render_markdown(report):
    lines = []
    summary = report["summary"]
    lines.append("# Human Tracks Data Engine Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Sequences scanned: {summary['n_sequences_scanned']}")
    lines.append(
        f"- With SAM3D predictions: {summary['n_sequences_with_sam3d_predictions']}"
    )
    lines.append(f"- With human tracks: {summary['n_sequences_with_human_tracks']}")
    lines.append(
        "- Total SAM3D predictions size: "
        f"{bytes_to_human(summary['total_sam3d_predictions_size_bytes'])}"
    )
    lines.append(
        "- Total human tracks size: "
        f"{bytes_to_human(summary['total_human_tracks_size_bytes'])}"
    )
    lines.append(
        "- Total generated size: "
        f"{bytes_to_human(summary['total_generated_size_bytes'])}"
    )
    if summary["mean_visibility_across_sequences"] is not None:
        lines.append(
            "- Mean visibility across sequences: "
            f"{summary['mean_visibility_across_sequences']:.3f}"
        )

    lines.append("")
    lines.append("## Per-Sequence")
    lines.append("")
    for seq in report["sequences"]:
        lines.append(f"### {seq['sequence']}")
        lines.append(f"- Path: `{seq['path']}`")
        lines.append(
            "- SAM3D predictions: "
            f"{bytes_to_human(seq['sam3d_predictions']['size_bytes'])} "
            f"({seq['sam3d_predictions']['n_prediction_files']} files)"
        )
        if seq["tracks"]["exists"]:
            tr = seq["tracks"]
            lines.append(
                "- Human tracks: "
                f"{bytes_to_human(tr['size_bytes'])}, "
                f"frames={tr.get('n_frames', 'n/a')}, views={tr.get('n_views', 'n/a')}, "
                f"persons={tr.get('n_persons', 'n/a')}, tracks={tr.get('n_tracks', 'n/a')}"
            )
            if "visibility_mean" in tr:
                lines.append(f"- Visibility mean: {tr['visibility_mean']:.3f}")
            if "valid_ratio" in tr:
                lines.append(f"- Valid ratio: {tr['valid_ratio']:.3f}")
            if "motion" in tr and "pelvis_jump_ratio" in tr["motion"]:
                lines.append(
                    f"- Pelvis jump ratio (> {tr['motion']['pelvis_jump_threshold']:.2f}m): "
                    f"{tr['motion']['pelvis_jump_ratio']:.3f}"
                )
            if "identity_ambiguous_ratio" in tr:
                lines.append(
                    f"- Identity ambiguous ratio: {tr['identity_ambiguous_ratio']:.3f}"
                )
        else:
            lines.append("- Human tracks: missing")

        if seq["issues"]:
            lines.append("- Issues:")
            for issue in seq["issues"]:
                lines.append(f"  - {issue}")
        if seq["recommendations"]:
            lines.append("- Recommendations:")
            for rec in seq["recommendations"]:
                lines.append(f"  - {rec}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def print_console_report(report):
    summary = report["summary"]
    print("=== Human Tracks Data Engine Report ===")
    print(f"Data root: {report['data_root']}")
    print(f"Sequences scanned: {summary['n_sequences_scanned']}")
    print(
        "Generated size: "
        f"{bytes_to_human(summary['total_generated_size_bytes'])} "
        f"(SAM3D {bytes_to_human(summary['total_sam3d_predictions_size_bytes'])} + "
        f"tracks {bytes_to_human(summary['total_human_tracks_size_bytes'])})"
    )
    if summary["mean_visibility_across_sequences"] is not None:
        print(f"Mean visibility across sequences: {summary['mean_visibility_across_sequences']:.3f}")
    print("")

    for seq in report["sequences"]:
        tracks = seq["tracks"]
        sam3d = seq["sam3d_predictions"]
        print(f"[{seq['sequence']}]")
        print(
            f"  SAM3D: {'OK' if sam3d['exists'] else 'MISSING'} | "
            f"views={sam3d.get('n_views_with_predictions', 0)} "
            f"files={sam3d.get('n_prediction_files', 0)} "
            f"size={bytes_to_human(sam3d.get('size_bytes', 0))}"
        )
        if tracks["exists"] and "visibility_mean" in tracks:
            motion = tracks.get("motion", {})
            print(
                f"  Tracks: OK | persons={tracks['n_persons']} tracks={tracks['n_tracks']} "
                f"frames={tracks['n_frames']} views={tracks['n_views']} "
                f"vis={tracks['visibility_mean']:.3f} valid={tracks['valid_ratio']:.3f} "
                f"jump_ratio={motion.get('pelvis_jump_ratio', 0.0):.3f} "
                f"size={bytes_to_human(tracks['size_bytes'])}"
            )
        else:
            print("  Tracks: MISSING")

        if seq["issues"]:
            print(f"  Issues: {', '.join(seq['issues'])}")
        print("")


def main():
    parser = argparse.ArgumentParser(
        description="Report generated human track data volume and quality metrics."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seq", type=str, default=None, help="Optional single sequence name")
    parser.add_argument(
        "--jump_threshold",
        type=float,
        default=0.5,
        help="Pelvis displacement threshold (meters/frame) for jump metric",
    )
    parser.add_argument("--save_json", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--save_markdown", type=str, default=None, help="Optional Markdown output path")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    sequences = list_sequences(args.data_root)
    if args.seq is not None:
        if args.seq not in sequences:
            raise ValueError(
                f"Sequence '{args.seq}' not found under {args.data_root}. Available: {sequences}"
            )
        sequences = [args.seq]

    seq_reports = []
    for seq_name in sequences:
        seq_path = os.path.join(args.data_root, seq_name)
        seq_reports.append(sequence_report(seq_name, seq_path, jump_threshold=args.jump_threshold))

    report = {
        "generated_at": datetime.now().isoformat(),
        "data_root": args.data_root,
        "summary": aggregate_report(seq_reports),
        "sequences": seq_reports,
    }

    print_console_report(report)

    if args.save_json:
        out_dir = os.path.dirname(args.save_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved JSON report to {args.save_json}")

    if args.save_markdown:
        out_dir = os.path.dirname(args.save_markdown)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.save_markdown, "w", encoding="utf-8") as f:
            f.write(render_markdown(report))
        print(f"Saved Markdown report to {args.save_markdown}")


if __name__ == "__main__":
    main()
