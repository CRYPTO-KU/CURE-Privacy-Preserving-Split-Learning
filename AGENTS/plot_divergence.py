#!/usr/bin/env python3
"""
Plot HE vs Plaintext Divergence Across Layers
Visualizes ACCUMULATED error from input through multi-layer neural networks
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from MNIST FC Model (784 → 128 → 32 → 10) - ACCUMULATED ERROR
mnist_fc = {
    "name": "MNIST FC (784→128→32→10)",
    "layers": [
        "Linear_0\n(784→128)", "Activation_0\n(ReLU3)", 
        "Linear_1\n(128→32)", "Activation_1\n(ReLU3)",
        "Linear_2\n(32→10)", "Activation_2\n(ReLU3)"
    ],
    "accumulated_rms": [1.98e-07, 1.40e-01, 1.93e-01, 1.92e-01, 9.78e-02, 1.95e-01],
    "accumulated_max": [4.44e-07, 2.38e-01, 3.61e-01, 2.50e-01, 1.79e-01, 2.42e-01],
    "layer_types": ["Linear", "Activation", "Linear", "Activation", "Linear", "Activation"],
    "total_rms": 3.76e-01
}

# Data from BCW FC Model (64 → 32 → 16 → 10) - ACCUMULATED ERROR
bcw_fc = {
    "name": "BCW FC (64→32→16→10)",
    "layers": [
        "Linear_0\n(64→32)", "Activation_0\n(ReLU3)", 
        "Linear_1\n(32→16)", "Activation_1\n(ReLU3)",
        "Linear_2\n(16→10)", "Activation_2\n(ReLU3)"
    ],
    "accumulated_rms": [4.85e-08, 1.33e-01, 2.11e-01, 2.08e-01, 1.16e-01, 2.17e-01],
    "accumulated_max": [1.30e-07, 2.50e-01, 4.05e-01, 2.44e-01, 1.90e-01, 2.49e-01],
    "layer_types": ["Linear", "Activation", "Linear", "Activation", "Linear", "Activation"],
    "total_rms": 4.07e-01
}

# Data from Small FC Model (8 → 4 → 2) - ACCUMULATED ERROR
small_fc = {
    "name": "Small FC (8→4→2)",
    "layers": [
        "Linear_0\n(8→4)", "Activation_0\n(ReLU3)", 
        "Linear_1\n(4→2)", "Activation_1\n(ReLU3)"
    ],
    "accumulated_rms": [1.01e-08, 9.74e-02, 1.29e-01, 2.07e-01],
    "accumulated_max": [1.26e-08, 1.14e-01, 1.73e-01, 2.41e-01],
    "layer_types": ["Linear", "Activation", "Linear", "Activation"],
    "total_rms": 2.62e-01
}

def plot_accumulated_error(data, ax):
    """Plot accumulated error from input at each layer"""
    x = np.arange(len(data["layers"]))
    
    # Colors for layer types
    colors = ['#2ecc71' if t == "Linear" else '#e74c3c' for t in data["layer_types"]]
    
    # Bar chart of accumulated RMS Error at each layer
    bars = ax.bar(x, data["accumulated_rms"], color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(data["layers"], fontsize=8)
    ax.set_ylabel('Accumulated RMS Error (from input)', fontsize=10)
    ax.set_title(f'{data["name"]}\nAccumulated Error at Each Layer', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1e-6, color='gray', linestyle='--', alpha=0.5, label='10⁻⁶ threshold')
    
    # Add value labels on bars
    for bar, val in zip(bars, data["accumulated_rms"]):
        height = bar.get_height()
        ax.annotate(f'{val:.2e}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=45)

def plot_comparison_all_models():
    """Create comparison plots for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HE vs Plaintext: Accumulated Error Analysis\n(Error measured from original input at each layer output)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Plot MNIST FC
    plot_accumulated_error(mnist_fc, axes[0, 0])
    
    # Plot BCW FC
    plot_accumulated_error(bcw_fc, axes[0, 1])
    
    # Plot Small FC
    plot_accumulated_error(small_fc, axes[1, 0])
    
    # Summary comparison plot
    ax = axes[1, 1]
    models = ['MNIST FC\n(3 layers)', 'BCW FC\n(3 layers)', 'Small FC\n(2 layers)']
    total_errors = [mnist_fc["total_rms"], bcw_fc["total_rms"], small_fc["total_rms"]]
    first_act_errors = [mnist_fc["accumulated_rms"][1], bcw_fc["accumulated_rms"][1], small_fc["accumulated_rms"][1]]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, first_act_errors, width, label='After 1st Activation', 
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, total_errors, width, label='Final Output', 
                   color='#9b59b6', alpha=0.8)
    
    ax.set_ylabel('Accumulated RMS Error', fontsize=10)
    ax.set_title('Model Comparison: Accumulated Error', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, first_act_errors):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar, val in zip(bars2, total_errors):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Add legend for layer types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Linear Layer (HE noise only ~10⁻⁸)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Activation Layer (Polynomial approx ~0.1-0.2)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
               bbox_to_anchor=(0.5, 0.98), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('divergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig('divergence_analysis.pdf', bbox_inches='tight')
    print("Saved: divergence_analysis.png and divergence_analysis.pdf")
    plt.show()

def plot_error_progression():
    """Show how accumulated error grows through layers"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot accumulated error progression for each model
    for data, color, marker in [
        (mnist_fc, '#3498db', 'o'), 
        (bcw_fc, '#e74c3c', 's'), 
        (small_fc, '#27ae60', '^')
    ]:
        x = np.arange(len(data["layers"]))
        ax.plot(x, data["accumulated_rms"], f'{marker}-', color=color, 
                linewidth=2.5, markersize=10, label=data["name"], alpha=0.8)
        ax.fill_between(x, 0, data["accumulated_rms"], alpha=0.1, color=color)
    
    # Shade layer types
    max_layers = max(len(mnist_fc["layers"]), len(bcw_fc["layers"]), len(small_fc["layers"]))
    for i in range(max_layers):
        if i % 2 == 0:  # Linear
            ax.axvspan(i-0.4, i+0.4, alpha=0.08, color='green')
        else:  # Activation
            ax.axvspan(i-0.4, i+0.4, alpha=0.08, color='red')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Accumulated RMS Error from Input', fontsize=12)
    ax.set_title('Error Accumulation Through Network Layers\n(Green = Linear, Red = Activation)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(6))
    ax.set_xticklabels(['Linear₀', 'Activ₀', 'Linear₁', 'Activ₁', 'Linear₂', 'Activ₂'])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(1e-9, 1)
    
    # Add annotations
    ax.annotate('Linear: ~10⁻⁸\n(pure HE noise)', 
                xy=(0, 5e-8), fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.annotate('1st Activation:\n~0.10-0.14 RMS\n(poly approx dominates)', 
                xy=(1, 0.2), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('error_progression.png', dpi=150, bbox_inches='tight')
    print("Saved: error_progression.png")
    plt.show()

def plot_layer_type_impact():
    """Compare the impact of linear vs activation layers on accumulated error"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collect error jumps (difference between consecutive layers)
    all_data = [mnist_fc, bcw_fc, small_fc]
    linear_initial = []  # First linear layer error (from zero)
    activation_jumps = []  # Error increase at activation layers
    linear_changes = []  # Error change at linear layers (after first)
    
    for data in all_data:
        for i, (rms, ltype) in enumerate(zip(data["accumulated_rms"], data["layer_types"])):
            if i == 0:  # First linear
                linear_initial.append(rms)
            elif ltype == "Activation":
                if i > 0:
                    jump = rms - data["accumulated_rms"][i-1]
                    activation_jumps.append(abs(jump))
            else:  # Subsequent linear
                if i > 0:
                    change = rms - data["accumulated_rms"][i-1]
                    linear_changes.append(change)
    
    # Box plot 1: Error by layer type
    bp = ax1.boxplot([linear_initial + linear_changes, activation_jumps], 
                     labels=['Linear Layers', 'Activation Layers'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax1.set_ylabel('Accumulated RMS Error', fontsize=11)
    ax1.set_title('Error Contribution by Layer Type', fontsize=12, fontweight='bold')
    ax1.set_yscale('symlog', linthresh=1e-6)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Error breakdown pie chart
    avg_linear_error = np.mean([mnist_fc["accumulated_rms"][0], bcw_fc["accumulated_rms"][0], small_fc["accumulated_rms"][0]])
    avg_first_act_error = np.mean([mnist_fc["accumulated_rms"][1], bcw_fc["accumulated_rms"][1], small_fc["accumulated_rms"][1]])
    avg_final_error = np.mean([mnist_fc["total_rms"], bcw_fc["total_rms"], small_fc["total_rms"]])
    
    # Estimate error sources
    he_noise_contribution = avg_linear_error / avg_final_error * 100
    poly_approx_contribution = (avg_first_act_error - avg_linear_error) / avg_final_error * 100
    propagation_contribution = (avg_final_error - avg_first_act_error) / avg_final_error * 100
    
    sizes = [he_noise_contribution, poly_approx_contribution, propagation_contribution]
    labels = [f'HE Noise\n({he_noise_contribution:.1e}%)', 
              f'Poly Approx\n({poly_approx_contribution:.1f}%)', 
              f'Propagation\n({propagation_contribution:.1f}%)']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax2.set_title('Error Source Breakdown\n(Approximate)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('error_breakdown.png', dpi=150, bbox_inches='tight')
    print("Saved: error_breakdown.png")
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("HE vs Plaintext: Accumulated Error Visualization")
    print("=" * 60)
    
    # Generate all plots
    print("\n1. Generating main comparison plots...")
    plot_comparison_all_models()
    
    print("\n2. Generating error progression...")
    plot_error_progression()
    
    print("\n3. Generating layer type impact analysis...")
    plot_layer_type_impact()
    
    print("\n" + "=" * 60)
    print("All plots saved!")
    print("=" * 60)
