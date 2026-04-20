#!/usr/bin/env python3
"""
Generate diagram showing depth-weighted multi-view fusion for mesh vertices.
Shows: (1) Mesh vertex projection to 4 views
       (2) Depth consistency check and weight computation
       (3) Weighted feature aggregation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as mlines

fig = plt.figure(figsize=(16, 9))

# Create custom layout with 3 main sections
ax_main = fig.add_subplot(111)
ax_main.set_xlim(0, 20)
ax_main.set_ylim(0, 12)
ax_main.axis('off')

# === Title ===
ax_main.text(10, 11.5, 'Depth-Weighted Multi-View Fusion', ha='center',
             fontsize=16, fontweight='bold')

# === Section 1: Mesh Vertex in 3D World Space ===
x_mesh, y_mesh = 2, 6
ax_main.add_patch(Circle((x_mesh, y_mesh), 0.5, facecolor='magenta', edgecolor='darkmagenta', lw=4, alpha=0.9))
ax_main.text(x_mesh, y_mesh, 'v', ha='center', va='center', fontsize=20, fontweight='bold', color='white')
ax_main.text(x_mesh, y_mesh - 1.2, 'Mesh Vertex\n(world space)', ha='center', fontsize=12, fontweight='bold')

# === Section 2: Projection to 4 Camera Views (laid out vertically with more space) ===
view_positions = [(6, 9.5), (6, 7), (6, 4.5), (6, 2)]
view_names = ['View 1', 'View 7', 'View 14', 'View 20']

# Depth consistency examples (simulated)
# View 1: visible (z_map ≈ z_proj → high weight)
# View 7: visible (z_map ≈ z_proj → high weight)
# View 14: occluded (z_map << z_proj → low weight)
# View 20: partially occluded (moderate mismatch → medium weight)
depth_scenarios = [
    {'z_proj': 3.5, 'z_map': 3.4, 'status': 'Visible', 'color': 'green'},
    {'z_proj': 3.8, 'z_map': 3.7, 'status': 'Visible', 'color': 'green'},
    {'z_proj': 4.2, 'z_map': 2.1, 'status': 'Occluded', 'color': 'red'},
    {'z_proj': 3.6, 'z_map': 3.0, 'status': 'Partial', 'color': 'orange'},
]

# Compute weights
weights = []
for scenario in depth_scenarios:
    z_proj = scenario['z_proj']
    z_map = scenario['z_map']
    alpha_d = 15.0
    ratio = z_map / z_proj
    w = np.exp(-alpha_d * abs(1.0 - ratio))
    weights.append(w)
    scenario['weight'] = w

# Normalize for display
weights_normalized = np.array(weights) / sum(weights)

# Draw as a clean table-like layout
col_widths = [2.5, 3.5, 2, 2]  # Camera, Depth Check, Weight, Feature
col_x = [5, 8.5, 12.5, 15]

# Column headers
headers = ['Camera\nView', 'Depth\nConsistency', 'Weight', 'Feature']
for x, header in zip(col_x, headers):
    ax_main.text(x, 10.5, header, ha='center', va='center',
                 fontsize=13, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', edgecolor='black', lw=2))

for i, (pos, name, scenario, w_norm) in enumerate(zip(view_positions, view_names, depth_scenarios, weights_normalized)):
    x_view, y_view = pos

    # Arrow from mesh to view (cleaner, single color)
    ax_main.annotate('', xy=(col_x[0] - 1.0, y_view), xytext=(x_mesh + 0.6, y_mesh),
                     arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.6))

    # Column 1: Camera view
    ax_main.add_patch(Rectangle((col_x[0] - 0.8, y_view - 0.4), 1.6, 0.8,
                                 facecolor='lightblue', edgecolor='black', lw=2))
    ax_main.text(col_x[0], y_view, name, ha='center', va='center', fontsize=12, fontweight='bold')

    # Column 2: Depth check
    box_color = scenario['color']
    ax_main.add_patch(FancyBboxPatch((col_x[1] - 1.5, y_view - 0.5), 3.0, 1.0,
                                     boxstyle="round,pad=0.1",
                                     facecolor='white', edgecolor=box_color, lw=3))

    ax_main.text(col_x[1], y_view + 0.25, f'z_proj = {scenario["z_proj"]:.1f}',
                 ha='center', fontsize=11, fontweight='bold')
    ax_main.text(col_x[1], y_view - 0.25, f'z_map = {scenario["z_map"]:.1f}',
                 ha='center', fontsize=11, fontweight='bold', color=box_color)

    # Status label
    ax_main.text(col_x[1], y_view - 0.8, scenario['status'], ha='center',
                fontsize=9, style='italic', color=box_color)

    # Column 3: Weight
    ax_main.add_patch(Circle((col_x[2], y_view), 0.5,
                            facecolor=scenario['color'], edgecolor='black', lw=3,
                            alpha=0.4 + 0.5*w_norm))
    ax_main.text(col_x[2], y_view, f'{scenario["weight"]:.2f}',
                 ha='center', va='center', fontsize=13, fontweight='bold')

    # Column 4: Feature vector (height indicates weight)
    feat_height = 0.4 + 1.0 * w_norm
    ax_main.add_patch(Rectangle((col_x[3] - 0.4, y_view - feat_height/2), 0.8, feat_height,
                                facecolor='purple', edgecolor='black', lw=2, alpha=0.8))
    ax_main.text(col_x[3], y_view, f'f_{i+1}', ha='center', va='center',
                 fontsize=13, fontweight='bold', color='white')

# === Section 3: Weighted Aggregation ===
x_agg = 17.5
y_agg = 6

# Arrows from features to aggregation (cleaner, bundled)
for i, (pos, w_norm) in enumerate(zip(view_positions, weights_normalized)):
    y_feat = pos[1]
    # Thicker arrows for higher weights
    ax_main.annotate('', xy=(x_agg - 0.9, y_agg), xytext=(col_x[3] + 0.5, y_feat),
                     arrowprops=dict(arrowstyle='->', lw=1.5 + 4*w_norm,
                                   color='purple', alpha=0.5 + 0.4*w_norm))

# Aggregation box
ax_main.add_patch(FancyBboxPatch((x_agg - 0.8, y_agg - 0.8), 1.6, 1.6,
                                boxstyle="round,pad=0.1",
                                facecolor='darkmagenta', edgecolor='black', lw=4))
ax_main.text(x_agg, y_agg + 0.3, 'Σ w·f', ha='center', va='center',
             fontsize=16, fontweight='bold', color='white')
ax_main.text(x_agg, y_agg, '─────', ha='center', va='center',
             fontsize=14, fontweight='bold', color='white')
ax_main.text(x_agg, y_agg - 0.3, 'Σ w', ha='center', va='center',
             fontsize=16, fontweight='bold', color='white')

ax_main.text(x_agg, y_agg - 1.5, 'Weighted\nAggregation', ha='center',
             fontsize=11, fontweight='bold')

# Final feature (larger, more prominent)
x_final = x_agg
y_final = 3
ax_main.annotate('', xy=(x_final, y_final + 0.7), xytext=(x_agg, y_agg - 0.9),
                 arrowprops=dict(arrowstyle='->', lw=4, color='darkmagenta'))

ax_main.add_patch(Rectangle((x_final - 0.6, y_final - 0.6), 1.2, 1.2,
                            facecolor='magenta', edgecolor='black', lw=4))
ax_main.text(x_final, y_final, 'f(v)', ha='center', va='center',
             fontsize=16, fontweight='bold', color='white')
ax_main.text(x_final, y_final - 1.5, 'Final 128-d\nFeature Vector', ha='center',
             fontsize=11, fontweight='bold')

# === Add formula box (larger, more prominent) ===
formula_text = r'$w_{\mathrm{depth}}(v) = \exp\left(-\alpha_d \left|1 - \frac{z_{\mathrm{map}}}{z_{\mathrm{proj}}}\right|\right),\quad \alpha_d = 15.0$'
ax_main.text(10, 0.8, formula_text, ha='center', fontsize=13,
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                      edgecolor='orange', lw=3))

# === Add legend in corner ===
legend_y = 10.8
ax_main.add_patch(Circle((1, legend_y), 0.15, facecolor='green', edgecolor='black', lw=1.5))
ax_main.text(1.5, legend_y, 'Depth point (~57k)', ha='left', va='center', fontsize=11)

ax_main.add_patch(Circle((1, legend_y - 0.6), 0.15, facecolor='magenta', edgecolor='black', lw=1.5))
ax_main.text(1.5, legend_y - 0.6, 'Mesh vertex (~7k)', ha='left', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('thesis/images/depth_weighted_fusion_diagram.pdf', bbox_inches='tight', dpi=300)
plt.savefig('thesis/images/depth_weighted_fusion_diagram.png', bbox_inches='tight', dpi=300)
print('Saved: thesis/images/depth_weighted_fusion_diagram.{pdf,png}')

# Print summary
print('\n=== Depth-Weighted Fusion Summary ===')
for i, (name, scenario, w) in enumerate(zip(view_names, depth_scenarios, weights)):
    print(f'{name}: z_proj={scenario["z_proj"]:.1f}, z_map={scenario["z_map"]:.1f} → w={w:.3f} ({scenario["status"]})')
print(f'\nNormalized weights: {[f"{w:.2f}" for w in weights_normalized]}')
print(f'Effective contributions: View 1 & 7 dominate (visible), View 14 suppressed (occluded)')
