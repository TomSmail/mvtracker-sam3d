#!/usr/bin/env python3
"""
Generate Figure 7: Depth contamination illustration showing performance inflation.
Shows: (1) How Dynamic3DGS uses all 31 views vs DUSt3R uses 4 views
       (2) 6x performance difference in baseline tracking
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel 1: Depth Estimation Pipelines ---
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('(a) Depth Estimation: All-View vs Target-View', fontsize=14, fontweight='bold')

# Dynamic3DGS pipeline
y_d3dgs = 7.5
ax1.add_patch(FancyBboxPatch((0.5, y_d3dgs-0.4), 2, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='red', facecolor='mistyrose', lw=2))
ax1.text(1.5, y_d3dgs, '31 Cameras\n(All Views)', ha='center', va='center', fontsize=10, fontweight='bold')

ax1.annotate('', xy=(3.2, y_d3dgs), xytext=(2.7, y_d3dgs),
             arrowprops=dict(arrowstyle='->', lw=2, color='red'))

ax1.add_patch(FancyBboxPatch((3.2, y_d3dgs-0.4), 2.5, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='red', facecolor='mistyrose', lw=2))
ax1.text(4.45, y_d3dgs, 'Multi-View\nOptimization', ha='center', va='center', fontsize=10, fontweight='bold')

ax1.annotate('', xy=(6.4, y_d3dgs), xytext=(5.9, y_d3dgs),
             arrowprops=dict(arrowstyle='->', lw=2, color='red'))

ax1.add_patch(FancyBboxPatch((6.4, y_d3dgs-0.4), 2.5, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='red', facecolor='lightcoral', lw=3))
ax1.text(7.65, y_d3dgs, 'Dynamic3DGS\nDepth Maps', ha='center', va='center', fontsize=10, fontweight='bold')

ax1.text(7.65, y_d3dgs-0.9, 'Near-perfect geometry\n(uses test views!)', ha='center', fontsize=8,
         color='darkred', style='italic')

# DUSt3R pipeline
y_duster = 4.5
ax1.add_patch(FancyBboxPatch((0.5, y_duster-0.4), 2, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='green', facecolor='honeydew', lw=2))
ax1.text(1.5, y_duster, '4 Cameras\n(Target Views)', ha='center', va='center', fontsize=10, fontweight='bold')

ax1.annotate('', xy=(3.2, y_duster), xytext=(2.7, y_duster),
             arrowprops=dict(arrowstyle='->', lw=2, color='green'))

ax1.add_patch(FancyBboxPatch((3.2, y_duster-0.4), 2.5, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='green', facecolor='honeydew', lw=2))
ax1.text(4.45, y_duster, 'ViT-Large\nRegression', ha='center', va='center', fontsize=10, fontweight='bold')

ax1.annotate('', xy=(6.4, y_duster), xytext=(5.9, y_duster),
             arrowprops=dict(arrowstyle='->', lw=2, color='green'))

ax1.add_patch(FancyBboxPatch((6.4, y_duster-0.4), 2.5, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='green', facecolor='lightgreen', lw=3))
ax1.text(7.65, y_duster, 'DUSt3R\nDepth Maps', ha='center', va='center', fontsize=10, fontweight='bold')

ax1.text(7.65, y_duster-0.9, 'Estimated, noisy\n(online-computable)', ha='center', fontsize=8,
         color='darkgreen', style='italic')

ax1.text(5, 2.5, 'Evaluation Question: Does SAM3D mesh help?', ha='center', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='orange', lw=2))

# --- Panel 2: Performance comparison ---
ax2 = axes[1]

depth_sources = ['Dynamic3DGS\n(all 31 views)', 'DUSt3R\n(4 target views)', 'Zero Depth\n(ablation)']
baseline_scores = [55.63, 9.07, 12.57]
best_sam3d_scores = [56.14, 8.81, 12.58]  # best for each depth type

x_pos = np.arange(len(depth_sources))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, baseline_scores, width, label='Baseline (no SAM3D)',
                color='steelblue', alpha=0.8, edgecolor='black', lw=1.5)
bars2 = ax2.bar(x_pos + width/2, best_sam3d_scores, width, label='Best SAM3D Variant',
                color='forestgreen', alpha=0.8, edgecolor='black', lw=1.5)

# Annotate the values
for i, (b, s) in enumerate(zip(baseline_scores, best_sam3d_scores)):
    ax2.text(i - width/2, b + 1, f'{b:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.text(i + width/2, s + 1, f'{s:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Show delta
    delta = s - b
    color = 'green' if delta > 0.3 else 'red' if delta < -0.3 else 'gray'
    ax2.text(i, max(b, s) + 4, f'{delta:+.1f}', ha='center', fontsize=11,
             color=color, fontweight='bold')

# Highlight the 6x inflation
ax2.annotate('', xy=(0, 55), xytext=(1, 9),
             arrowprops=dict(arrowstyle='<->', lw=3, color='red', linestyle='--'))
ax2.text(0.5, 32, '6× inflation\nfrom depth\ncontamination', ha='center', fontsize=11,
         color='darkred', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='red', lw=2))

ax2.set_ylabel('3D Average Jaccard (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Depth Source', fontsize=12, fontweight='bold')
ax2.set_title('(b) Performance by Depth Quality (Averaged Across 3 View Configs)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(depth_sources, fontsize=11)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 65)

plt.tight_layout()
plt.savefig('thesis/images/depth_contamination.pdf', bbox_inches='tight', dpi=300)
plt.savefig('thesis/images/depth_contamination.png', bbox_inches='tight', dpi=300)
print('Saved: thesis/images/depth_contamination.{pdf,png}')

# Print summary
print(f'\n=== Summary ===')
print(f'Performance inflation from Dynamic3DGS: {baseline_scores[0] / baseline_scores[1]:.1f}x')
print(f'Depth-weighted improvement on DUSt3R: {best_sam3d_scores[1] - baseline_scores[1]:.2f} AJ points')
print(f'Depth-weighted improvement on Dynamic3DGS: {best_sam3d_scores[0] - baseline_scores[0]:.2f} AJ points')
