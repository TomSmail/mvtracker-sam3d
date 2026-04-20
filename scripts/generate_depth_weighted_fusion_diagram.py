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

fig = plt.figure(figsize=(16, 10))

# Create custom layout with 3 main sections
ax_main = fig.add_subplot(111)
ax_main.set_xlim(0, 16)
ax_main.set_ylim(0, 10)
ax_main.axis('off')

# === Section 1: Mesh Vertex in 3D World Space ===
x_mesh, y_mesh = 2, 7
ax_main.add_patch(Circle((x_mesh, y_mesh), 0.3, facecolor='magenta', edgecolor='darkmagenta', lw=3, alpha=0.8))
ax_main.text(x_mesh, y_mesh, 'v', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
ax_main.text(x_mesh, y_mesh - 0.8, 'Mesh Vertex\n(world space)', ha='center', fontsize=10, fontweight='bold')

# === Section 2: Projection to 4 Camera Views ===
view_positions = [(5, 8.5), (5, 6.5), (5, 4.5), (5, 2.5)]
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

ax_main.text(5, 9.5, 'Multi-View Projection', ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black', lw=2))

for i, (pos, name, scenario, w_norm) in enumerate(zip(view_positions, view_names, depth_scenarios, weights_normalized)):
    x_view, y_view = pos

    # Draw camera icon
    ax_main.add_patch(Rectangle((x_view - 0.4, y_view - 0.25), 0.8, 0.5,
                                 facecolor='lightblue', edgecolor='black', lw=1.5))
    ax_main.text(x_view, y_view, name, ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow from mesh to view
    arrow_color = scenario['color']
    arrow_alpha = 0.3 + 0.5 * w_norm  # Thicker/more opaque for higher weight
    ax_main.annotate('', xy=(x_view - 0.5, y_view), xytext=(x_mesh + 0.3, y_mesh),
                     arrowprops=dict(arrowstyle='->', lw=2 + 3*w_norm, color=arrow_color, alpha=arrow_alpha))

    # Depth check box
    x_check = x_view + 2.5
    box_color = scenario['color']
    ax_main.add_patch(FancyBboxPatch((x_check - 1.2, y_view - 0.35), 2.4, 0.7,
                                     boxstyle="round,pad=0.05",
                                     facecolor='white', edgecolor=box_color, lw=2))

    ax_main.text(x_check, y_view + 0.15, f'z_proj={scenario["z_proj"]:.1f}',
                 ha='center', fontsize=8)
    ax_main.text(x_check, y_view - 0.15, f'z_map={scenario["z_map"]:.1f}',
                 ha='center', fontsize=8, color=box_color)

    # Weight computation
    x_weight = x_check + 2.2
    ax_main.add_patch(Circle((x_weight, y_view), 0.35,
                            facecolor=scenario['color'], edgecolor='black', lw=2,
                            alpha=0.3 + 0.6*w_norm))
    ax_main.text(x_weight, y_view, f'w={scenario["weight"]:.2f}',
                 ha='center', va='center', fontsize=9, fontweight='bold')

    # Feature vector (size indicates weight)
    x_feat = x_weight + 1.8
    feat_height = 0.3 + 0.6 * w_norm
    ax_main.add_patch(Rectangle((x_feat - 0.3, y_view - feat_height/2), 0.6, feat_height,
                                facecolor='purple', edgecolor='black', lw=1.5, alpha=0.7))
    ax_main.text(x_feat, y_view, f'f_{i+1}', ha='center', va='center',
                 fontsize=10, fontweight='bold', color='white')

# === Section 3: Weighted Aggregation ===
x_agg = 13
y_agg = 5.5

# Draw weighted sum
ax_main.text(x_agg, 8, 'Weighted Aggregation', ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black', lw=2))

# Arrows from features to aggregation
for i, (pos, w_norm) in enumerate(zip(view_positions, weights_normalized)):
    x_feat = 12
    y_feat = pos[1]
    ax_main.annotate('', xy=(x_agg - 0.5, y_agg), xytext=(x_feat, y_feat),
                     arrowprops=dict(arrowstyle='->', lw=1 + 3*w_norm,
                                   color='purple', alpha=0.4 + 0.5*w_norm))

# Aggregation circle
ax_main.add_patch(Circle((x_agg, y_agg), 0.6,
                        facecolor='darkmagenta', edgecolor='black', lw=3))
ax_main.text(x_agg, y_agg + 0.1, 'Σ w·f', ha='center', va='center',
             fontsize=14, fontweight='bold', color='white')
ax_main.text(x_agg, y_agg - 0.35, '─────', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white')
ax_main.text(x_agg, y_agg - 0.6, 'Σ w', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white')

# Final feature
x_final = x_agg + 2
ax_main.add_patch(Rectangle((x_final - 0.4, y_agg - 0.4), 0.8, 0.8,
                            facecolor='magenta', edgecolor='black', lw=3))
ax_main.text(x_final, y_agg, 'f(v)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white')
ax_main.text(x_final, y_agg - 1.2, 'Final 128-d\nFeature Vector', ha='center', fontsize=10, fontweight='bold')

ax_main.annotate('', xy=(x_final - 0.5, y_agg), xytext=(x_agg + 0.6, y_agg),
                 arrowprops=dict(arrowstyle='->', lw=3, color='darkmagenta'))

# === Add formula box ===
formula_text = r'$w_{\mathrm{depth}}(v) = \exp\left(-\alpha_d \left|1 - \frac{z_{\mathrm{map}}}{z_{\mathrm{proj}}}\right|\right)$' + '\n' + r'$\alpha_d = 15.0$'
ax_main.text(8, 0.8, formula_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                      edgecolor='orange', lw=2.5))

# === Add status legend ===
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                  markersize=10, label='Visible (w ≈ 1.0)', markeredgecolor='black'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                  markersize=10, label='Partial (w ≈ 0.5)', markeredgecolor='black'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=10, label='Occluded (w ≈ 0.0)', markeredgecolor='black'),
]
ax_main.legend(handles=legend_elements, loc='lower left', fontsize=10,
              framealpha=0.95, title='Depth Consistency', title_fontsize=11)

# === Add title ===
ax_main.text(8, 10.5, 'Depth-Weighted Multi-View Fusion for Mesh Vertex Features',
             ha='center', fontsize=14, fontweight='bold')

# === Add point cloud legend ===
ax_main.add_patch(Circle((0.8, 9.5), 0.12, facecolor='green', edgecolor='black', lw=1))
ax_main.text(1.2, 9.5, 'Depth point (~57k)', ha='left', va='center', fontsize=9)

ax_main.add_patch(Circle((0.8, 9.0), 0.12, facecolor='magenta', edgecolor='black', lw=1))
ax_main.text(1.2, 9.0, 'Mesh vertex (~7k)', ha='left', va='center', fontsize=9)

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
