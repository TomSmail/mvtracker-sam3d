#!/usr/bin/env python3
"""
Generate Figure 6: Comparison grid showing AJ across depth types and methods.
3x3 grid: rows = depth sources (Dynamic3DGS, DUSt3R, Zero), cols = methods (Baseline, Mean, Depth-Weighted)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns

# Data from all evaluations (Average Jaccard across 3 view configs)
data = {
    'Dynamic3DGS': {
        'Baseline': 55.63,
        'Mean Fusion': 56.14,
        'Depth-Weighted': 56.14,  # Identical to mean with perfect depth
    },
    'DUSt3R': {
        'Baseline': 9.07,
        'Mean Fusion': 7.93,
        'Depth-Weighted': 8.81,
    },
    'Zero Depth': {
        'Baseline': 12.57,
        'Mean Fusion': 12.58,
        'Depth-Weighted': 12.58,  # No benefit without depth
    }
}

# Per-view-config breakdown for DUSt3R (where fusion matters)
per_config_duster = {
    'views1_7_14_20': {'Baseline': 11.56, 'Mean': 9.67, 'Depth-Wt': 9.67},
    'views27_16_14_8': {'Baseline': 7.90, 'Mean': 7.31, 'Depth-Wt': 7.31},
    'views1_4_7_11': {'Baseline': 7.76, 'Mean': 6.81, 'Depth-Wt': 9.45},  # THE WIN
}

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# --- Panel 1: Heatmap ---
ax1 = axes[0]
depth_types = ['Dynamic3DGS\n(near-perfect)', 'DUSt3R\n(realistic)', 'Zero Depth\n(ablation)']
methods = ['Baseline', 'Mean Fusion', 'Depth-Weighted']

matrix = np.array([
    [data['Dynamic3DGS']['Baseline'], data['Dynamic3DGS']['Mean Fusion'], data['Dynamic3DGS']['Depth-Weighted']],
    [data['DUSt3R']['Baseline'], data['DUSt3R']['Mean Fusion'], data['DUSt3R']['Depth-Weighted']],
    [data['Zero Depth']['Baseline'], data['Zero Depth']['Mean Fusion'], data['Zero Depth']['Depth-Weighted']],
])

# Use log scale coloring since Dynamic3DGS values dominate
sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=60,
            xticklabels=methods, yticklabels=depth_types, ax=ax1,
            cbar_kws={'label': '3D Average Jaccard (%)'},
            linewidths=0.5, linecolor='gray')

ax1.set_title('(a) Average Jaccard Across Depth Sources and Fusion Strategies', fontsize=14, fontweight='bold')
ax1.set_xlabel('')
ax1.set_ylabel('')

# Annotate key cells
# Best with perfect depth
ax1.add_patch(mpatches.Rectangle((1, 0), 1, 1, fill=False, edgecolor='blue', lw=3))
ax1.text(1.5, 0.2, '↑ Best with\nperfect depth', ha='center', fontsize=8, color='blue', fontweight='bold')

# Best with realistic depth
ax1.add_patch(mpatches.Rectangle((2, 1), 1, 1, fill=False, edgecolor='darkgreen', lw=3))
ax1.text(2.5, 1.8, '↑ Only positive\nresult', ha='center', fontsize=8, color='darkgreen', fontweight='bold')

# --- Panel 2: Per-view-config breakdown for DUSt3R ---
ax2 = axes[1]
configs = list(per_config_duster.keys())
x = np.arange(len(configs))
width = 0.25

baselines = [per_config_duster[c]['Baseline'] for c in configs]
means = [per_config_duster[c]['Mean'] for c in configs]
dws = [per_config_duster[c]['Depth-Wt'] for c in configs]

bars1 = ax2.bar(x - width, baselines, width, label='Baseline (no SAM3D)', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x, means, width, label='+ SAM3D (mean fusion)', color='coral', alpha=0.8)
bars3 = ax2.bar(x + width, dws, width, label='+ SAM3D (depth-weighted)', color='forestgreen', alpha=0.8)

# Annotate the win on views1_4_7_11
ax2.annotate('', xy=(2 + width, dws[2]), xytext=(2 - width, baselines[2]),
             arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
ax2.text(2, 10.5, f'+{dws[2] - baselines[2]:.1f} AJ\n-20mm ATE', ha='center', fontsize=9,
         color='darkgreen', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

ax2.set_xlabel('Camera Configuration', fontsize=12)
ax2.set_ylabel('3D Average Jaccard (%)', fontsize=12)
ax2.set_title('(b) DUSt3R Depth: Per-Configuration Breakdown', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['views 1,7,14,20\n(wide baseline)', 'views 27,16,14,8\n(wide baseline)',
                     'views 1,4,7,11\n(narrow baseline)'], fontsize=10)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 13)

plt.tight_layout()
plt.savefig('thesis/images/comparison_grid.pdf', bbox_inches='tight', dpi=300)
plt.savefig('thesis/images/comparison_grid.png', bbox_inches='tight', dpi=300)
print('Saved: thesis/images/comparison_grid.{pdf,png}')
plt.close()

print(f'\nKey findings:')
print(f'  Dynamic3DGS: Baseline {data["Dynamic3DGS"]["Baseline"]:.1f}% → +SAM3D {data["Dynamic3DGS"]["Mean Fusion"]:.1f}% (+{data["Dynamic3DGS"]["Mean Fusion"] - data["Dynamic3DGS"]["Baseline"]:.1f})')
print(f'  DUSt3R:      Baseline {data["DUSt3R"]["Baseline"]:.1f}% → depth-wt {data["DUSt3R"]["Depth-Weighted"]:.1f}% ({data["DUSt3R"]["Depth-Weighted"] - data["DUSt3R"]["Baseline"]:.1f})')
print(f'  Zero:        Baseline {data["Zero Depth"]["Baseline"]:.1f}% → +SAM3D {data["Zero Depth"]["Mean Fusion"]:.1f}% (+{data["Zero Depth"]["Mean Fusion"] - data["Zero Depth"]["Baseline"]:.2f})')
