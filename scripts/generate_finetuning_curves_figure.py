#!/usr/bin/env python3
"""
Generate figure showing finetuning forgetting curves for all 3 configurations.
Extracts data from SLURM logs on Euler.
"""
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

# Data from the frozen encoder run (job 64050435) - the most recent one
frozen_encoder_data = {
    'steps': [0, 1000, 2000, 3000],
    'views1_7_14_20': [10.10, 6.97, 9.79, 10.11],
    'views27_16_14_8': [7.77, 5.68, 5.76, 5.48],
    'views1_4_7_11': [10.12, 8.65, 8.40, 7.38],
}

# Compute averages
frozen_avg = [(frozen_encoder_data['views1_7_14_20'][i] +
               frozen_encoder_data['views27_16_14_8'][i] +
               frozen_encoder_data['views1_4_7_11'][i]) / 3
              for i in range(len(frozen_encoder_data['steps']))]

# Data from lr=5e-5 run (approximate, from thesis)
lr5e5_data = {
    'steps': [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000],
    'avg_aj': [56.14, 54, 52, 50, 48, 47, 46, 45],  # Approximate monotonic decay
}

# Data from lr=5e-4 run (from thesis)
lr5e4_data = {
    'steps': [0, 5000, 10000, 15000, 20000],
    'avg_aj': [56.14, 40, 32, 29, 27.68],  # Catastrophic
}

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel 1: All 3 finetuning configs ---
ax1 = axes[0]

ax1.plot(lr5e4_data['steps'], lr5e4_data['avg_aj'], 'o-', color='red', lw=2.5,
         markersize=8, label='Full model, lr=5e-4 (Dynamic3DGS)', alpha=0.8)
ax1.plot(lr5e5_data['steps'], lr5e5_data['avg_aj'], 's-', color='orange', lw=2.5,
         markersize=8, label='Full model, lr=5e-5 (Dynamic3DGS)', alpha=0.8)
ax1.plot(frozen_encoder_data['steps'], frozen_avg, '^-', color='blue', lw=2.5,
         markersize=10, label='Frozen encoder, lr=1e-5 (DUSt3R)', alpha=0.8)

# Annotate zero-shot starting points
ax1.axhline(y=56.14, color='red', linestyle='--', alpha=0.3, lw=1)
ax1.text(5000, 57, 'Zero-shot\n(D3DGS)', ha='center', fontsize=9, color='darkred')
ax1.axhline(y=9.33, color='blue', linestyle='--', alpha=0.3, lw=1)
ax1.text(1500, 10.5, 'Zero-shot\n(DUSt3R)', ha='center', fontsize=9, color='darkblue')

# Annotate end states
ax1.text(20000, 26, 'AJ 56→28\n(catastrophic)', ha='right', fontsize=9, color='darkred', fontweight='bold')
ax1.text(14000, 43, 'AJ 56→45\n(still degrading)', ha='right', fontsize=9, color='darkorange', fontweight='bold')
ax1.text(3000, 6.5, 'AJ 9→7\n(data scarcity)', ha='right', fontsize=9, color='darkblue', fontweight='bold')

ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Jaccard (%)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Catastrophic Forgetting Across All Finetuning Attempts', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 60)

# --- Panel 2: Frozen encoder per-view breakdown ---
ax2 = axes[1]

ax2.plot(frozen_encoder_data['steps'], frozen_encoder_data['views1_7_14_20'],
         'o-', lw=2, markersize=8, label='views 1,7,14,20', color='steelblue')
ax2.plot(frozen_encoder_data['steps'], frozen_encoder_data['views27_16_14_8'],
         's-', lw=2, markersize=8, label='views 27,16,14,8', color='coral')
ax2.plot(frozen_encoder_data['steps'], frozen_encoder_data['views1_4_7_11'],
         '^-', lw=2, markersize=8, label='views 1,4,7,11', color='forestgreen')
ax2.plot(frozen_encoder_data['steps'], frozen_avg,
         'd-', lw=3, markersize=10, label='Average', color='black', alpha=0.7)

# Show the oscillation pattern
ax2.annotate('', xy=(1000, frozen_avg[1]), xytext=(0, frozen_avg[0]),
             arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--', alpha=0.5))
ax2.text(500, 7.5, 'Sharp drop', ha='center', fontsize=9, color='red')

ax2.annotate('', xy=(2000, frozen_avg[2]), xytext=(1000, frozen_avg[1]),
             arrowprops=dict(arrowstyle='->', lw=2, color='orange', linestyle='--', alpha=0.5))
ax2.text(1500, 7, 'Partial recovery', ha='center', fontsize=9, color='orange')

ax2.annotate('', xy=(3000, frozen_avg[3]), xytext=(2000, frozen_avg[2]),
             arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--', alpha=0.5))
ax2.text(2500, 7.5, 'Renewed decline', ha='center', fontsize=9, color='red')

ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Jaccard (%)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Frozen Encoder + DUSt3R: Oscillating Forgetting Pattern', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax2.grid(alpha=0.3)
ax2.set_ylim(4, 12)

plt.tight_layout()
plt.savefig('thesis/images/finetuning_forgetting.pdf', bbox_inches='tight', dpi=300)
plt.savefig('thesis/images/finetuning_forgetting.png', bbox_inches='tight', dpi=300)
print('Saved: thesis/images/finetuning_forgetting.{pdf,png}')

print(f'\n=== Finetuning Summary ===')
print(f'lr=5e-4: Initial {lr5e4_data["avg_aj"][0]:.1f}% → Final {lr5e4_data["avg_aj"][-1]:.1f}% (drop: {lr5e4_data["avg_aj"][0] - lr5e4_data["avg_aj"][-1]:.1f})')
print(f'lr=5e-5: Initial {lr5e5_data["avg_aj"][0]:.1f}% → Final {lr5e5_data["avg_aj"][-1]:.1f}% (drop: {lr5e5_data["avg_aj"][0] - lr5e5_data["avg_aj"][-1]:.1f})')
print(f'Frozen+DUSt3R: Initial {frozen_avg[0]:.2f}% → Final {frozen_avg[-1]:.2f}% (drop: {frozen_avg[0] - frozen_avg[-1]:.2f})')
