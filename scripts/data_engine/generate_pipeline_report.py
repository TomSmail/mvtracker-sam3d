"""
Generate a comprehensive visual report of the SAM3D data pipeline output.

Produces a standalone report.html (base64-embedded images) covering:
  1. Pipeline status overview (processed vs failed, by category)
  2. Dataset coverage (views, frames, sequence types)
  3. Track quality metrics (visibility, valid ratio, n_persons)
  4. Track volume statistics
  5. Motion quality indicators
  6. Sample frame visualizations from the 6 best sequences
  7. Full sequence table with colour-coded quality flags

Usage:
    python scripts/data_engine/generate_pipeline_report.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
        --output_dir reports/pipeline_report
"""

import argparse
import base64
import io
import os
import re
import sys
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.data_engine.report_human_tracks import summarize_tracks_file, list_sequences
from scripts.data_engine.visualize_tracks import render_tracks_frame

# ── colour palette ────────────────────────────────────────────────────────────
C_GREEN  = "#2ecc71"
C_AMBER  = "#f39c12"
C_RED    = "#e74c3c"
C_BLUE   = "#3498db"
C_GREY   = "#95a5a6"
C_PURPLE = "#9b59b6"
C_TEAL   = "#1abc9c"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
})


# ── sequence classification ───────────────────────────────────────────────────

def classify_sequence(name):
    """Return a human-readable activity category from the sequence directory name."""
    patterns = [
        (r"haggling",   "Haggling"),
        (r"mafia",      "Mafia"),
        (r"ultimatum",  "Ultimatum"),
        (r"band",       "Band"),
        (r"ian",        "Ian"),
        (r"flute",      "Flute"),
        (r"office",     "Office"),
        (r"dance",      "Dance"),
        (r"moonbaby",   "Moonbaby"),
        (r"toddler",    "Toddler"),
        (r"cello",      "Cello"),
        (r"pose",       "Pose"),
        (r"basketball", "Sports"),
        (r"football",   "Sports"),
        (r"softball",   "Sports"),
        (r"tennis",     "Sports"),
        (r"boxes",      "Object"),
        (r"juggle",     "Juggle"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "Other"


def get_failure_reason(seq_path):
    """Determine why a sequence has no human_tracks.npz."""
    ims_path = os.path.join(seq_path, "ims")
    if not os.path.isdir(ims_path) or len(os.listdir(ims_path)) == 0:
        return "No images"
    ann_file = os.path.join(seq_path, "tapvid3d_annotations.npz")
    if not os.path.exists(ann_file):
        return "Missing annotations"
    sam3d_dir = os.path.join(seq_path, "sam3d_predictions")
    if not os.path.isdir(sam3d_dir):
        return "No SAM3D predictions"
    return "Other data issue"


# ── data loading ──────────────────────────────────────────────────────────────

def load_all_data(data_root, jump_threshold=0.5):
    """Return (processed_list, failed_list) where each item is a dict."""
    sequences = list_sequences(data_root)
    processed, failed = [], []
    for seq_name in sequences:
        seq_path = os.path.join(data_root, seq_name)
        tracks_file = os.path.join(seq_path, "human_tracks.npz")
        if os.path.exists(tracks_file):
            metrics = summarize_tracks_file(seq_path, jump_threshold=jump_threshold)
            metrics["name"] = seq_name
            metrics["category"] = classify_sequence(seq_name)
            processed.append(metrics)
        else:
            failed.append({
                "name": seq_name,
                "category": classify_sequence(seq_name),
                "reason": get_failure_reason(seq_path),
            })
    return processed, failed


# ── figure helpers ─────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def img_to_base64(img_bgr):
    """Convert a BGR numpy array to base64-encoded PNG."""
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Section 1: Pipeline Status ────────────────────────────────────────────────

def fig_pipeline_status(processed, failed):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Section 1 — Pipeline Status", fontsize=14, fontweight="bold")

    # Left: Pie chart overall
    reason_counts = {}
    for f in failed:
        reason_counts[f["reason"]] = reason_counts.get(f["reason"], 0) + 1
    labels = ["Processed"] + list(reason_counts.keys())
    sizes  = [len(processed)] + list(reason_counts.values())
    colors = [C_GREEN] + [C_RED, C_AMBER, C_GREY, C_PURPLE][:len(reason_counts)]
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(10)
    axes[0].set_title(f"Overall ({len(processed) + len(failed)} sequences)", fontweight="bold")

    # Right: Stacked bar by activity category
    all_categories = sorted(set(
        [p["category"] for p in processed] + [f["category"] for f in failed]
    ))
    proc_counts = {c: 0 for c in all_categories}
    fail_counts = {c: 0 for c in all_categories}
    for p in processed:
        proc_counts[p["category"]] += 1
    for f in failed:
        fail_counts[f["category"]] += 1

    x = np.arange(len(all_categories))
    bars_ok   = axes[1].bar(x, [proc_counts[c] for c in all_categories],
                             color=C_GREEN, label="Processed", edgecolor="white")
    bars_fail = axes[1].bar(x, [fail_counts[c] for c in all_categories],
                             bottom=[proc_counts[c] for c in all_categories],
                             color=C_RED, label="Failed/Skipped", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_categories, rotation=35, ha="right")
    axes[1].set_ylabel("Sequences")
    axes[1].set_title("Status by Activity Type", fontweight="bold")
    axes[1].legend()

    fig.tight_layout()
    return fig


# ── Section 2: Dataset Coverage ───────────────────────────────────────────────

def fig_dataset_coverage(processed):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Section 2 — Dataset Coverage", fontsize=14, fontweight="bold")

    n_views  = [p["n_views"]  for p in processed if "n_views"  in p]
    n_frames = [p["n_frames"] for p in processed if "n_frames" in p]

    # Views histogram
    axes[0].hist(n_views, bins=range(min(n_views), max(n_views) + 2),
                 color=C_BLUE, edgecolor="white", rwidth=0.8)
    axes[0].set_xlabel("Camera Views per Sequence")
    axes[0].set_ylabel("Sequences")
    axes[0].set_title("Views Distribution", fontweight="bold")
    axes[0].axvline(np.mean(n_views), color=C_RED, linestyle="--",
                    label=f"Mean {np.mean(n_views):.1f}")
    axes[0].legend()

    # Frames histogram
    axes[1].hist(n_frames, bins=20, color=C_TEAL, edgecolor="white")
    axes[1].set_xlabel("Frames per Sequence")
    axes[1].set_ylabel("Sequences")
    axes[1].set_title("Sequence Length Distribution", fontweight="bold")
    axes[1].axvline(np.mean(n_frames), color=C_RED, linestyle="--",
                    label=f"Mean {np.mean(n_frames):.0f}")
    axes[1].legend()

    # Scatter: views × frames
    n_tracks = [p["n_tracks"] for p in processed if "n_tracks" in p and "n_frames" in p and "n_views" in p]
    sc = axes[2].scatter(n_views, n_frames,
                          c=n_tracks, cmap="viridis",
                          s=60, alpha=0.8, edgecolors="none")
    plt.colorbar(sc, ax=axes[2], label="N tracks")
    axes[2].set_xlabel("Camera Views")
    axes[2].set_ylabel("Frames")
    axes[2].set_title("Views × Frames (colour = tracks)", fontweight="bold")

    fig.tight_layout()
    return fig


# ── Section 3: Track Quality ──────────────────────────────────────────────────

def fig_track_quality(processed):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Section 3 — Track Quality", fontsize=14, fontweight="bold")

    vis    = [p["visibility_mean"] for p in processed if "visibility_mean" in p]
    valid  = [p["valid_ratio"]     for p in processed if "valid_ratio"     in p]
    n_pers = [p["n_persons"]       for p in processed if "n_persons"       in p]

    # Visibility distribution
    axes[0, 0].hist(vis, bins=20, color=C_GREEN, edgecolor="white")
    axes[0, 0].axvline(np.mean(vis), color=C_RED, linestyle="--",
                       label=f"Mean {np.mean(vis):.3f}")
    axes[0, 0].axvline(0.45, color=C_AMBER, linestyle=":",
                       label="Warning threshold (0.45)")
    axes[0, 0].set_xlabel("Visibility Mean")
    axes[0, 0].set_ylabel("Sequences")
    axes[0, 0].set_title("Visibility (any-view, any-frame)", fontweight="bold")
    axes[0, 0].legend(fontsize=9)

    # Valid ratio distribution
    axes[0, 1].hist(valid, bins=20, color=C_BLUE, edgecolor="white")
    axes[0, 1].axvline(np.mean(valid), color=C_RED, linestyle="--",
                       label=f"Mean {np.mean(valid):.3f}")
    axes[0, 1].set_xlabel("Valid Track Ratio")
    axes[0, 1].set_ylabel("Sequences")
    axes[0, 1].set_title("Track Validity (fraction of T×N pairs valid)", fontweight="bold")
    axes[0, 1].legend(fontsize=9)

    # Scatter: visibility vs valid_ratio
    colors = [C_GREEN if v >= 0.45 else (C_AMBER if v >= 0.30 else C_RED) for v in vis]
    axes[1, 0].scatter(vis, valid, c=colors, s=50, alpha=0.7, edgecolors="none")
    axes[1, 0].set_xlabel("Visibility Mean")
    axes[1, 0].set_ylabel("Valid Ratio")
    axes[1, 0].set_title("Quality Cluster (green=OK, amber=warn, red=low)", fontweight="bold")
    axes[1, 0].axvline(0.45, color=C_AMBER, linestyle=":", alpha=0.5)

    # Persons per sequence
    from collections import Counter
    cnt = Counter(n_pers)
    bars = axes[1, 1].bar([str(k) for k in sorted(cnt)],
                           [cnt[k] for k in sorted(cnt)],
                           color=C_PURPLE, edgecolor="white")
    axes[1, 1].set_xlabel("Persons Detected")
    axes[1, 1].set_ylabel("Sequences")
    axes[1, 1].set_title("Persons per Sequence", fontweight="bold")
    for bar in bars:
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(int(bar.get_height())), ha="center", fontsize=10)

    fig.tight_layout()
    return fig


# ── Section 4: Track Volume ───────────────────────────────────────────────────

def fig_track_volume(processed):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Section 4 — Track Volume", fontsize=14, fontweight="bold")

    n_tracks = [p["n_tracks"] for p in processed if "n_tracks" in p]
    n_frames = [p["n_frames"] for p in processed if "n_frames" in p and "n_tracks" in p]
    n_kp     = [p["n_keypoints"]    for p in processed if "n_keypoints"    in p]
    n_verts  = [p["n_sampled_verts"] for p in processed if "n_sampled_verts" in p]
    n_pers   = [p["n_persons"]       for p in processed if "n_persons"       in p and
                "n_keypoints" in p and "n_sampled_verts" in p]

    # Tracks per sequence histogram
    axes[0].hist(n_tracks, bins=20, color=C_TEAL, edgecolor="white")
    axes[0].axvline(np.mean(n_tracks), color=C_RED, linestyle="--",
                    label=f"Mean {np.mean(n_tracks):.0f}")
    axes[0].set_xlabel("Tracks per Sequence")
    axes[0].set_ylabel("Sequences")
    axes[0].set_title("Tracks per Sequence", fontweight="bold")
    axes[0].legend()

    # Scatter: frames × tracks (dataset volume)
    axes[1].scatter(n_frames, n_tracks, color=C_BLUE, s=50, alpha=0.7, edgecolors="none")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Tracks")
    axes[1].set_title("Dataset Volume (Frames × Tracks)", fontweight="bold")
    # annotate with total data points
    total_pts = sum(f * n for f, n in zip(n_frames, n_tracks))
    axes[1].text(0.98, 0.02, f"Total: {total_pts:,} pts",
                 transform=axes[1].transAxes, ha="right", fontsize=9, color=C_GREY)

    # Keypoints vs vertices breakdown
    if n_pers and n_kp and n_verts:
        total_kp    = [p * k for p, k in zip(n_pers, n_kp)]
        total_verts = [p * v for p, v in zip(n_pers, n_verts)]
        names = [p["name"][:15] for p in processed if "n_persons" in p and "n_keypoints" in p]
        x = np.arange(len(names))
        axes[2].bar(x, total_kp,    color=C_BLUE,   label="Keypoints (70×persons)", edgecolor="none")
        axes[2].bar(x, total_verts, bottom=total_kp, color=C_GREEN,  label="Vertices (500×persons)", edgecolor="none")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=90, fontsize=6)
        axes[2].set_ylabel("Tracks")
        axes[2].set_title("Keypoints vs Vertices per Sequence", fontweight="bold")
        axes[2].legend(fontsize=8)

    fig.tight_layout()
    return fig


# ── Section 5: Motion Quality ─────────────────────────────────────────────────

def fig_motion_quality(processed):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Section 5 — Motion Quality", fontsize=14, fontweight="bold")

    has_motion = [p for p in processed if p.get("motion") and "pelvis_speed_mean" in p.get("motion", {})]

    if not has_motion:
        for ax in axes:
            ax.text(0.5, 0.5, "No motion data", ha="center", transform=ax.transAxes)
        return fig

    speeds    = [p["motion"]["pelvis_speed_mean"]  for p in has_motion]
    jump_rate = [p["motion"]["pelvis_jump_ratio"]  for p in has_motion]
    vis       = [p["visibility_mean"]              for p in has_motion if "visibility_mean" in p]

    # Pelvis speed distribution
    axes[0].hist(speeds, bins=15, color=C_BLUE, edgecolor="white")
    axes[0].axvline(np.mean(speeds), color=C_RED, linestyle="--",
                    label=f"Mean {np.mean(speeds):.3f} m/f")
    axes[0].set_xlabel("Pelvis Speed (m/frame)")
    axes[0].set_ylabel("Sequences")
    axes[0].set_title("Pelvis Speed Distribution", fontweight="bold")
    axes[0].legend()

    # Jump rate distribution
    axes[1].hist(jump_rate, bins=15, color=C_AMBER, edgecolor="white")
    axes[1].axvline(0.05, color=C_RED, linestyle=":", label="Warning (5%)")
    axes[1].axvline(np.mean(jump_rate), color=C_RED, linestyle="--",
                    label=f"Mean {np.mean(jump_rate):.3f}")
    axes[1].set_xlabel("Pelvis Jump Ratio (>0.5m/frame)")
    axes[1].set_ylabel("Sequences")
    axes[1].set_title("Temporal Discontinuity Rate\n(lower = smoother trajectories)", fontweight="bold")
    axes[1].legend(fontsize=9)

    # Quality scatter: visibility vs jump rate
    if vis:
        colors_q = [C_GREEN if jr < 0.05 else (C_AMBER if jr < 0.15 else C_RED) for jr in jump_rate]
        axes[2].scatter(vis[:len(jump_rate)], jump_rate,
                        c=colors_q, s=60, alpha=0.8, edgecolors="none")
        axes[2].axhline(0.05, color=C_AMBER, linestyle=":", alpha=0.5)
        axes[2].set_xlabel("Visibility Mean")
        axes[2].set_ylabel("Jump Ratio")
        axes[2].set_title("Visibility vs Jump Rate", fontweight="bold")

    fig.tight_layout()
    return fig


# ── Section 6: Sample Visualizations ─────────────────────────────────────────

def find_readable_view_and_frame(seq_path, tracks_data):
    """Return (view_idx, frame_idx) with highest visible track count at mid-sequence."""
    vis = tracks_data["per_view_visibilities"]   # [V, T, N]
    T = vis.shape[1]
    frame_candidates = [T // 4, T // 2, 3 * T // 4]
    best_view, best_frame, best_count = 0, T // 2, 0
    for frame_idx in frame_candidates:
        for view_idx in range(vis.shape[0]):
            ims_path = os.path.join(seq_path, "ims")
            view_dirs = sorted(os.listdir(ims_path), key=lambda x: int(x))
            if view_idx >= len(view_dirs):
                continue
            v_name = view_dirs[view_idx]
            img_files = sorted(os.listdir(os.path.join(ims_path, v_name)))
            if frame_idx >= len(img_files):
                continue
            img_path = os.path.join(ims_path, v_name, img_files[frame_idx])
            if os.path.getsize(img_path) == 0:
                continue
            count = int(vis[view_idx, frame_idx].sum())
            if count > best_count:
                best_count = count
                best_view = view_idx
                best_frame = frame_idx
    return best_view, best_frame


def render_sample(seq_path, seq_name, max_size=640):
    """Render one representative frame from a sequence. Returns BGR image or None."""
    tracks_file = os.path.join(seq_path, "human_tracks.npz")
    if not os.path.exists(tracks_file):
        return None
    try:
        with np.load(tracks_file, allow_pickle=True) as td:
            tracks_data = dict(td)
        view_idx, frame_idx = find_readable_view_and_frame(seq_path, tracks_data)

        ims_path = os.path.join(seq_path, "ims")
        view_dirs = sorted(os.listdir(ims_path), key=lambda x: int(x))
        v_name = view_dirs[view_idx]
        img_files = sorted(os.listdir(os.path.join(ims_path, v_name)))
        img_path = os.path.join(ims_path, v_name, img_files[frame_idx])
        img = cv2.imread(img_path)
        if img is None:
            return None
        img_rgb = img[:, :, ::-1].copy()

        rendered = render_tracks_frame(img_rgb, tracks_data, view_idx, frame_idx,
                                       show_vertices=True, show_trails=False)
        rendered_bgr = rendered[:, :, ::-1]

        # Resize if needed
        h, w = rendered_bgr.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            rendered_bgr = cv2.resize(rendered_bgr, (int(w * scale), int(h * scale)))
        return rendered_bgr
    except Exception as e:
        warnings.warn(f"Could not render sample for {seq_name}: {e}")
        return None


def render_four_view_grid(seq_path, seq_name):
    """Render the same frame from 4 views side by side for one sequence."""
    tracks_file = os.path.join(seq_path, "human_tracks.npz")
    if not os.path.exists(tracks_file):
        return None
    try:
        with np.load(tracks_file, allow_pickle=True) as td:
            tracks_data = dict(td)
        _, frame_idx = find_readable_view_and_frame(seq_path, tracks_data)

        ims_path = os.path.join(seq_path, "ims")
        view_dirs = sorted(os.listdir(ims_path), key=lambda x: int(x))
        n_views = min(4, len(view_dirs), int(tracks_data["per_view_visibilities"].shape[0]))
        panels = []
        for view_idx in range(n_views):
            v_name = view_dirs[view_idx]
            img_files = sorted(os.listdir(os.path.join(ims_path, v_name)))
            if frame_idx >= len(img_files):
                continue
            img_path = os.path.join(ims_path, v_name, img_files[frame_idx])
            if os.path.getsize(img_path) == 0:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = img[:, :, ::-1].copy()
            rendered = render_tracks_frame(img_rgb, tracks_data, view_idx, frame_idx,
                                           show_vertices=True, show_trails=False)
            # Add view label
            rendered_bgr = rendered[:, :, ::-1].copy()
            cv2.putText(rendered_bgr, f"View {view_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # Scale to common height
            h, w = rendered_bgr.shape[:2]
            target_h = 320
            rendered_bgr = cv2.resize(rendered_bgr,
                                       (int(w * target_h / h), target_h))
            panels.append(rendered_bgr)
        if not panels:
            return None
        return np.concatenate(panels, axis=1)
    except Exception as e:
        warnings.warn(f"Could not render 4-view grid for {seq_name}: {e}")
        return None


def generate_sample_visualizations(processed, data_root, n_samples=6):
    """Return list of (seq_name, base64_png) for top-quality sequences."""
    # Sort by visibility mean, pick top n_samples
    ranked = sorted(
        [p for p in processed if "visibility_mean" in p],
        key=lambda x: x["visibility_mean"],
        reverse=True,
    )[:n_samples]

    results = []
    for p in ranked:
        seq_path = os.path.join(data_root, p["name"])
        img = render_sample(seq_path, p["name"])
        if img is not None:
            results.append((p["name"], p["visibility_mean"], img_to_base64(img)))

    # 4-view grid for the single best sequence
    grid_b64 = None
    grid_name = None
    if ranked:
        best = ranked[0]
        seq_path = os.path.join(data_root, best["name"])
        grid_img = render_four_view_grid(seq_path, best["name"])
        if grid_img is not None:
            grid_b64 = img_to_base64(grid_img)
            grid_name = best["name"]

    return results, grid_b64, grid_name


# ── Section 7: Per-view visibility box plot ────────────────────────────────────

def fig_per_view_visibility(processed):
    """Box plot of per-view visibility across all processed sequences."""
    all_per_view = [p["visibility_per_view_mean"]
                    for p in processed if "visibility_per_view_mean" in p]
    if not all_per_view:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
        return fig

    max_views = max(len(v) for v in all_per_view)
    data_by_view = [
        [v[i] for v in all_per_view if i < len(v)]
        for i in range(max_views)
    ]

    fig, ax = plt.subplots(figsize=(max(6, max_views * 0.6), 5))
    bp = ax.boxplot(data_by_view, patch_artist=True, notch=False,
                    medianprops={"color": C_RED, "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BLUE)
        patch.set_alpha(0.6)
    ax.set_xlabel("Camera View Index")
    ax.set_ylabel("Visibility Mean")
    ax.set_title("Per-View Visibility Distribution Across Sequences", fontweight="bold")
    ax.axhline(0.45, color=C_AMBER, linestyle=":", alpha=0.7, label="Warning threshold")
    ax.legend()
    fig.tight_layout()
    return fig


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html(status_b64, coverage_b64, quality_b64, volume_b64, motion_b64,
               per_view_b64, samples, grid_b64, grid_name, processed, failed):

    n_proc = len(processed)
    n_fail = len(failed)
    n_total = n_proc + n_fail

    vis_values = [p["visibility_mean"] for p in processed if "visibility_mean" in p]
    valid_values = [p["valid_ratio"] for p in processed if "valid_ratio" in p]
    total_tracks = sum(p.get("n_tracks", 0) * p.get("n_frames", 0) for p in processed)

    def row_colour(p):
        vis = p.get("visibility_mean", 0)
        vr  = p.get("valid_ratio", 0)
        jr  = p.get("motion", {}).get("pelvis_jump_ratio", 0)
        if vis < 0.30 or jr > 0.15:
            return "#fde8e8"
        if vis < 0.45 or vr < 0.99 or jr > 0.05:
            return "#fef9e7"
        return "#eafaf1"

    table_rows = []
    for p in sorted(processed, key=lambda x: x.get("visibility_mean", 0), reverse=True):
        bg = row_colour(p)
        m = p.get("motion", {})
        ambig = f'{p.get("identity_ambiguous_ratio", 0):.2f}' if "identity_ambiguous_ratio" in p else "—"
        table_rows.append(f"""
        <tr style="background:{bg}">
          <td>{p["name"]}</td>
          <td>{p["category"]}</td>
          <td>{p.get("n_persons","—")}</td>
          <td>{p.get("n_views","—")}</td>
          <td>{p.get("n_frames","—")}</td>
          <td>{p.get("n_tracks","—")}</td>
          <td>{p.get("visibility_mean", 0):.3f}</td>
          <td>{p.get("valid_ratio", 0):.3f}</td>
          <td>{m.get("pelvis_speed_mean", 0):.3f}</td>
          <td>{m.get("pelvis_jump_ratio", 0):.3f}</td>
          <td>{ambig}</td>
        </tr>""")

    fail_rows = "\n".join(
        f'<tr style="background:#fde8e8"><td>{f["name"]}</td><td>{f["category"]}</td>'
        f'<td colspan="9">{f["reason"]}</td></tr>'
        for f in failed
    )

    sample_cards = ""
    for seq_name, vis_val, b64 in samples:
        sample_cards += f"""
        <div class="card">
          <img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px">
          <p><b>{seq_name}</b> &nbsp;|&nbsp; visibility {vis_val:.3f}</p>
        </div>"""

    grid_section = ""
    if grid_b64 and grid_name:
        grid_section = f"""
        <h3>4-Camera-View Grid: {grid_name}</h3>
        <img src="data:image/png;base64,{grid_b64}"
             style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15)">"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAM3D Data Pipeline Report</title>
<style>
  body  {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 24px; color: #2c3e50; }}
  h1   {{ border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
  h2   {{ color: #2980b9; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 40px; }}
  h3   {{ color: #34495e; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .stat-box  {{ background: #f0f8ff; border-radius: 10px; padding: 16px; text-align: center;
                border: 1px solid #d0e8f0; }}
  .stat-box  .val {{ font-size: 2em; font-weight: 700; color: #2980b9; }}
  .stat-box  .lbl {{ color: #7f8c8d; font-size: 0.85em; margin-top: 4px; }}
  figure     {{ margin: 0; }}
  figure img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.12); }}
  .card-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px; }}
  .card      {{ background: #f8f9fa; border-radius: 8px; padding: 10px; text-align: center;
                font-size: 0.85em; border: 1px solid #e0e0e0; }}
  table      {{ border-collapse: collapse; width: 100%; font-size: 0.82em; }}
  th         {{ background: #2980b9; color: white; padding: 8px 10px; text-align: left; }}
  td         {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
  .legend    {{ display: flex; gap: 16px; margin: 8px 0; font-size: 0.85em; }}
  .dot       {{ width: 14px; height: 14px; border-radius: 50%; display: inline-block;
                margin-right: 5px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>SAM3D Data Pipeline Report</h1>
<p style="color:#7f8c8d">Panoptic Studio · SAM3D human mesh priors · Generated automatically</p>

<div class="stat-grid">
  <div class="stat-box"><div class="val">{n_proc}</div><div class="lbl">Sequences Processed</div></div>
  <div class="stat-box"><div class="val">{n_fail}</div><div class="lbl">Skipped / Failed</div></div>
  <div class="stat-box"><div class="val">{np.mean(vis_values):.2%}</div><div class="lbl">Mean Visibility</div></div>
  <div class="stat-box"><div class="val">{total_tracks:,}</div><div class="lbl">Total Track-Frames</div></div>
</div>

<h2>1 · Pipeline Status</h2>
<figure><img src="data:image/png;base64,{status_b64}"></figure>

<h2>2 · Dataset Coverage</h2>
<figure><img src="data:image/png;base64,{coverage_b64}"></figure>

<h2>3 · Track Quality</h2>
<figure><img src="data:image/png;base64,{quality_b64}"></figure>
<figure><img src="data:image/png;base64,{per_view_b64}"></figure>

<h2>4 · Track Volume</h2>
<figure><img src="data:image/png;base64,{volume_b64}"></figure>

<h2>5 · Motion Quality</h2>
<figure><img src="data:image/png;base64,{motion_b64}"></figure>

<h2>6 · Sample Visualisations</h2>
<p>Top sequences by visibility mean. Skeleton = MHR-70 joints; small dots = sampled mesh vertices.</p>
<div class="card-grid">{sample_cards}</div>
{grid_section}

<h2>7 · Full Sequence Table</h2>
<div class="legend">
  <span><span class="dot" style="background:#c6efce"></span>Good (vis ≥ 0.45, jump &lt; 5%)</span>
  <span><span class="dot" style="background:#ffeb9c"></span>Warning</span>
  <span><span class="dot" style="background:#ffc7ce"></span>Issue</span>
</div>
<table>
  <thead><tr>
    <th>Sequence</th><th>Category</th><th>Persons</th><th>Views</th><th>Frames</th>
    <th>Tracks</th><th>Visibility</th><th>Valid%</th>
    <th>Speed(m/f)</th><th>Jump%</th><th>Ambig%</th>
  </tr></thead>
  <tbody>
  {''.join(table_rows)}
  <tr><td colspan="11" style="background:#fde8e8;font-weight:bold">
      ── Sequences not processed ({n_fail}) ──</td></tr>
  {fail_rows}
  </tbody>
</table>
</body>
</html>
"""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate visual data pipeline report")
    parser.add_argument("--data_root", required=True,
                        help="Root of panoptic-multiview dataset")
    parser.add_argument("--output_dir", default="reports/pipeline_report",
                        help="Output directory for report files")
    parser.add_argument("--n_samples", type=int, default=6,
                        help="Number of sample visualisations to render")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figs_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    print("Loading metrics for all sequences...")
    processed, failed = load_all_data(args.data_root)
    print(f"  Processed: {len(processed)}  |  Failed/skipped: {len(failed)}")

    print("Generating charts...")
    fig1 = fig_pipeline_status(processed, failed)
    fig2 = fig_dataset_coverage(processed)
    fig3 = fig_track_quality(processed)
    fig4 = fig_track_volume(processed)
    fig5 = fig_motion_quality(processed)
    fig6 = fig_per_view_visibility(processed)

    # Save individual figures
    for name, fig in [("status", fig1), ("coverage", fig2), ("quality", fig3),
                       ("volume", fig4), ("motion", fig5), ("per_view", fig6)]:
        fig.savefig(os.path.join(figs_dir, f"{name}.png"), dpi=130, bbox_inches="tight")
        print(f"  Saved {name}.png")

    # Base64 encode for HTML embedding
    b64_status    = fig_to_base64(fig1)
    b64_coverage  = fig_to_base64(fig2)
    b64_quality   = fig_to_base64(fig3)
    b64_volume    = fig_to_base64(fig4)
    b64_motion    = fig_to_base64(fig5)
    b64_per_view  = fig_to_base64(fig6)

    print(f"Rendering {args.n_samples} sample visualisations...")
    samples, grid_b64, grid_name = generate_sample_visualizations(
        processed, args.data_root, n_samples=args.n_samples
    )
    print(f"  Rendered {len(samples)} samples")

    print("Building HTML report...")
    html = build_html(
        b64_status, b64_coverage, b64_quality, b64_volume, b64_motion, b64_per_view,
        samples, grid_b64, grid_name, processed, failed,
    )
    out_path = os.path.join(args.output_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to: {out_path}")
    print(f"Figures saved to: {figs_dir}/")


if __name__ == "__main__":
    main()
