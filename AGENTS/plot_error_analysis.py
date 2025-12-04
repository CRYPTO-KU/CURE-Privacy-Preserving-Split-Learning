#!/usr/bin/env python3
"""
HE vs Plaintext Error Analysis Visualization

This script visualizes the error analysis results from CURE_lib,
showing how error accumulates through neural network layers.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Data from test results (after scale bug fix)
# Format: (layer_name, rms_error)

data = {
    'MNIST_MLP': {
        'architecture': [784, 128, 32, 10],
        'total_rms': 4.786544e-01,
        'layers': [
            ('Linear_0 (784→128)', 1.980002e-07),
            ('Activation_0 (ReLU3)', 2.013526e-01),
            ('Linear_1 (128→32)', 1.933889e-01),
            ('Activation_1 (ReLU3)', 2.615387e-01),
            ('Linear_2 (32→10)', 9.781349e-02),
            ('Activation_2 (ReLU3)', 2.705511e-01),
        ]
    },
    'BCW_FC': {
        'architecture': [64, 32, 16, 10],
        'total_rms': 5.052641e-01,
        'layers': [
            ('Linear_0 (64→32)', 4.022058e-08),
            ('Activation_0 (ReLU3)', 1.916088e-01),
            ('Linear_1 (32→16)', 2.114868e-01),
            ('Activation_1 (ReLU3)', 2.722934e-01),
            ('Linear_2 (16→10)', 1.163549e-01),
            ('Activation_2 (ReLU3)', 2.935456e-01),
        ]
    },
    'LeNet_FC': {
        'architecture': [256, 120, 84, 10],
        'total_rms': 4.984516e-01,
        'layers': [
            ('Linear_0 (256→120)', 9.608939e-08),
            ('Activation_0 (ReLU3)', 2.083677e-01),
            ('Linear_1 (120→84)', 1.905846e-01),
            ('Activation_1 (ReLU3)', 2.677572e-01),
            ('Linear_2 (84→10)', 1.332838e-01),
            ('Activation_2 (ReLU3)', 2.815243e-01),
        ]
    },
    'Audio1D_FC': {
        'architecture': [2000, 5],
        'total_rms': 2.040206e-01,
        'layers': [
            ('Linear_0 (2000→5)', 2.264650e-07),
            ('Activation_0 (ReLU3)', 2.040206e-01),
        ]
    },
    'Small_FC': {
        'architecture': [16, 8, 4, 2],
        'total_rms': 4.743749e-01,
        'layers': [
            ('Linear_0 (16→8)', 2.985600e-08),
            ('Activation_0 (ReLU3)', 2.440817e-01),
            ('Linear_1 (8→4)', 9.891413e-02),
            ('Activation_1 (ReLU3)', 2.792326e-01),
            ('Linear_2 (4→2)', 5.715435e-02),
            ('Activation_2 (ReLU3)', 2.728263e-01),
        ]
    },
}

# Create output directory
output_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Figure 1: Layer-by-layer RMS error for all models
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

colors_linear = '#2196F3'  # Blue for Linear
colors_activation = '#F44336'  # Red for Activation

for idx, (model_name, model_data) in enumerate(data.items()):
    ax = axes[idx]
    layers = model_data['layers']
    
    x = np.arange(len(layers))
    rms_values = [l[1] for l in layers]
    layer_names = [l[0].split(' ')[0] for l in layers]
    
    # Color bars based on layer type
    colors = [colors_linear if 'Linear' in l[0] else colors_activation for l in layers]
    
    bars = ax.bar(x, rms_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yscale('log')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('RMS Error (log scale)')
    ax.set_title(f'{model_name}\nArchitecture: {model_data["architecture"]}')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.axhline(y=1e-7, color='green', linestyle='--', alpha=0.5, label='HE Noise Floor (~1e-7)')
    ax.axhline(y=0.18, color='orange', linestyle='--', alpha=0.5, label='Poly Approx Error (~0.18)')
    ax.set_ylim(1e-9, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars, rms_values):
        height = bar.get_height()
        if height > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., height*1.5,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height*1.5,
                   f'{height:.1e}', ha='center', va='bottom', fontsize=8)

# Hide the last empty subplot
axes[5].axis('off')

# Add legend to the last subplot area
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_linear, edgecolor='black', label='Linear Layer'),
    Patch(facecolor=colors_activation, edgecolor='black', label='Activation (ReLU3)'),
    plt.Line2D([0], [0], color='green', linestyle='--', label='HE Noise Floor (~1e-7)'),
    plt.Line2D([0], [0], color='orange', linestyle='--', label='Poly Approx Error (~0.18)'),
]
axes[5].legend(handles=legend_elements, loc='center', fontsize=14)
axes[5].set_title('Legend', fontsize=14)

plt.suptitle('HE vs Plaintext Divergence: Layer-by-Layer RMS Error\n(Comparing HE Polynomial to True ReLU)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'layer_by_layer_error.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(plots_dir, 'layer_by_layer_error.pdf'), bbox_inches='tight')
print(f"Saved: {os.path.join(plots_dir, 'layer_by_layer_error.png')}")

# Figure 2: Error breakdown - Linear vs Activation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Extract linear and activation errors for each model
models = list(data.keys())
linear_errors = []
activation_errors = []

for model_name in models:
    layers = data[model_name]['layers']
    linear_err = np.mean([l[1] for l in layers if 'Linear' in l[0]])
    activation_err = np.mean([l[1] for l in layers if 'Activation' in l[0]])
    linear_errors.append(linear_err)
    activation_errors.append(activation_err)

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, linear_errors, width, label='Linear Layers', color=colors_linear, alpha=0.8)
bars2 = ax1.bar(x + width/2, activation_errors, width, label='Activation Layers', color=colors_activation, alpha=0.8)

ax1.set_yscale('log')
ax1.set_ylabel('Mean RMS Error (log scale)')
ax1.set_xlabel('Model')
ax1.set_title('Mean Error by Layer Type')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend()
ax1.set_ylim(1e-9, 1)

# Add value annotations
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height*1.5,
            f'{height:.1e}', ha='center', va='bottom', fontsize=8, rotation=90)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height*1.5,
            f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

# Total RMS error comparison
total_errors = [data[m]['total_rms'] for m in models]
bars3 = ax2.bar(x, total_errors, color='#9C27B0', alpha=0.8, edgecolor='black')
ax2.set_ylabel('Total Accumulated RMS Error')
ax2.set_xlabel('Model')
ax2.set_title('Total Forward Pass Error')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')

# Add value annotations
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.suptitle('Error Analysis Summary: Linear Layer Noise vs Activation Approximation Error',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'error_breakdown.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(plots_dir, 'error_breakdown.pdf'), bbox_inches='tight')
print(f"Saved: {os.path.join(plots_dir, 'error_breakdown.png')}")

# Figure 3: Error source visualization (pie chart)
fig, ax = plt.subplots(figsize=(10, 8))

# Average across all models
all_linear_errors = []
all_activation_errors = []
for model_data in data.values():
    for layer_name, rms in model_data['layers']:
        if 'Linear' in layer_name:
            all_linear_errors.append(rms)
        else:
            all_activation_errors.append(rms)

avg_linear = np.mean(all_linear_errors)
avg_activation = np.mean(all_activation_errors)

# Since linear error is so small, we'll show it as a comparison
error_sources = ['HE Computation Noise\n(Linear Layers)', 'Polynomial Approximation\n(Activation Layers)']
errors = [avg_linear, avg_activation]
colors_pie = [colors_linear, colors_activation]

# Create a bar chart instead (pie chart would be misleading due to scale difference)
bars = ax.barh(error_sources, errors, color=colors_pie, alpha=0.8, edgecolor='black')
ax.set_xscale('log')
ax.set_xlabel('Mean RMS Error (log scale)')
ax.set_title('Error Source Comparison\n(Averaged Across All Models)', fontsize=14, fontweight='bold')
ax.set_xlim(1e-9, 1)

# Add value annotations
for bar, val in zip(bars, errors):
    ax.text(val * 2, bar.get_y() + bar.get_height()/2,
           f'{val:.2e}', ha='left', va='center', fontsize=12)

# Add annotation explaining the difference
ax.annotate('', xy=(avg_activation, 0.5), xytext=(avg_linear, 0.5),
           arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ratio = avg_activation / avg_linear
ax.text(np.sqrt(avg_linear * avg_activation), 0.7, 
       f'Ratio: {ratio:.0e}x', ha='center', fontsize=12, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'error_sources.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(plots_dir, 'error_sources.pdf'), bbox_inches='tight')
print(f"Saved: {os.path.join(plots_dir, 'error_sources.png')}")

# Figure 4: Key insight visualization
fig, ax = plt.subplots(figsize=(12, 6))

# Show what happens at each layer for Small_FC model
model = 'Small_FC'
layers = data[model]['layers']
x = range(len(layers))
rms_values = [l[1] for l in layers]
layer_names = [l[0] for l in layers]

# Plot with different markers for linear vs activation
for i, (name, rms) in enumerate(layers):
    if 'Linear' in name:
        ax.scatter(i, rms, s=200, c=colors_linear, marker='s', edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(f'HE Noise\n~{rms:.0e}', (i, rms), textcoords="offset points", 
                   xytext=(0, 20), ha='center', fontsize=10, color=colors_linear)
    else:
        ax.scatter(i, rms, s=200, c=colors_activation, marker='o', edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(f'Poly Error\n~{rms:.2f}', (i, rms), textcoords="offset points", 
                   xytext=(0, 20), ha='center', fontsize=10, color=colors_activation)

# Connect the dots
ax.plot(x, rms_values, 'k--', alpha=0.3, linewidth=2)

ax.set_yscale('log')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('RMS Error (log scale)', fontsize=12)
ax.set_title(f'Key Insight: Why Error "Blows Up" After First Activation\n({model}: {data[model]["architecture"]})',
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(layer_names, rotation=30, ha='right')
ax.set_ylim(1e-9, 1)

# Add explanation box
explanation = """
KEY FINDING:
• Linear layers: Pure HE noise (~1e-8)
• Activation layers: Polynomial approximation error (~0.2)

The "error explosion" after activation is NOT HE noise accumulation,
but the inherent difference between ReLU3 polynomial and true ReLU:
  ReLU3(x) = 0.318 + 0.5x + 0.212x² ≠ max(0, x)
"""
ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=10,
       verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors_linear, 
           markersize=15, markeredgecolor='black', markeredgewidth=2, label='Linear (HE Noise)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_activation, 
           markersize=15, markeredgecolor='black', markeredgewidth=2, label='Activation (Poly Approx)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'key_insight.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(plots_dir, 'key_insight.pdf'), bbox_inches='tight')
print(f"Saved: {os.path.join(plots_dir, 'key_insight.png')}")

print("\n✅ All plots generated successfully!")
print(f"📁 Output directory: {plots_dir}")
