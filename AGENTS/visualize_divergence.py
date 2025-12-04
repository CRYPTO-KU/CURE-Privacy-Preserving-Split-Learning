#!/usr/bin/env python3
"""
HE Divergence Visualization Script
Generates plots for layer-by-layer error analysis from HE vs Plaintext tests.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# DATA: Layer-by-layer results from Go tests (Updated Nov 29, 2025)
# ============================================================================

# MNIST MLP (784 → 128 → 32 → 10)
mnist_mlp_data = {
    'layer_idx': [0, 1, 2, 3, 4, 5],
    'layer_name': [
        'Linear_0\n(784→128)', 'ReLU3_0', 
        'Linear_1\n(128→32)', 'ReLU3_1', 
        'Linear_2\n(32→10)', 'ReLU3_2'
    ],
    'layer_type': ['Linear', 'Activation', 'Linear', 'Activation', 'Linear', 'Activation'],
    'max_abs_err': [4.48e-07, 2.38e-01, 3.61e-01, 2.50e-01, 1.79e-01, 2.42e-01],
    'mean_abs_err': [1.29e-07, 1.20e-01, 1.61e-01, 1.76e-01, 7.85e-02, 1.85e-01],
    'rms_err': [1.63e-07, 1.40e-01, 1.93e-01, 1.92e-01, 9.78e-02, 1.95e-01],
    'max_rel_err': [3.74e-05, 5.50e+01, 5.44e+00, 6.01e+01, 1.65e+00, 7.14e+00],
    'dims': [128, 128, 32, 32, 10, 10]
}

# BCW FC (64 → 32 → 16 → 10)
bcw_fc_data = {
    'layer_idx': [0, 1, 2, 3, 4, 5],
    'layer_name': [
        'Linear_0\n(64→32)', 'ReLU3_0', 
        'Linear_1\n(32→16)', 'ReLU3_1', 
        'Linear_2\n(16→10)', 'ReLU3_2'
    ],
    'layer_type': ['Linear', 'Activation', 'Linear', 'Activation', 'Linear', 'Activation'],
    'max_abs_err': [1.31e-07, 2.50e-01, 4.05e-01, 2.44e-01, 1.90e-01, 2.49e-01],
    'mean_abs_err': [3.34e-08, 1.15e-01, 1.70e-01, 1.99e-01, 1.02e-01, 2.11e-01],
    'rms_err': [4.56e-08, 1.33e-01, 2.11e-01, 2.08e-01, 1.16e-01, 2.17e-01],
    'max_rel_err': [9.85e-07, 1.10e+01, 1.98e+00, 6.34e+00, 1.32e+00, 9.56e-01],
    'dims': [32, 32, 16, 16, 10, 10]
}

# LeNet FC (256 → 120 → 84 → 10)
lenet_fc_data = {
    'layer_idx': [0, 1, 2, 3, 4, 5],
    'layer_name': [
        'Linear_0\n(256→120)', 'ReLU3_0', 
        'Linear_1\n(120→84)', 'ReLU3_1', 
        'Linear_2\n(84→10)', 'ReLU3_2'
    ],
    'layer_type': ['Linear', 'Activation', 'Linear', 'Activation', 'Linear', 'Activation'],
    'max_abs_err': [2.26e-07, 2.36e-01, 4.25e-01, 2.58e-01, 3.09e-01, 2.56e-01],
    'mean_abs_err': [7.17e-08, 1.27e-01, 1.56e-01, 1.78e-01, 1.04e-01, 1.94e-01],
    'rms_err': [8.93e-08, 1.45e-01, 1.91e-01, 1.97e-01, 1.33e-01, 2.06e-01],
    'max_rel_err': [1.48e-05, 6.07e+01, 1.73e+01, 9.82e+01, 2.12e+00, 5.94e+00],
    'dims': [120, 120, 84, 84, 10, 10]
}

# Audio1D FC (2000 → 5)
audio1d_fc_data = {
    'layer_idx': [0, 1],
    'layer_name': ['Linear_0\n(2000→5)', 'ReLU3_0'],
    'layer_type': ['Linear', 'Activation'],
    'max_abs_err': [4.87e-07, 2.01e-01],
    'mean_abs_err': [2.29e-07, 1.14e-01],
    'rms_err': [2.96e-07, 1.36e-01],
    'max_rel_err': [2.80e-06, 7.69e-01],
    'dims': [5, 5]
}

# Small FC (16 → 8 → 4 → 2)
small_fc_data = {
    'layer_idx': [0, 1, 2, 3, 4, 5],
    'layer_name': [
        'Linear_0\n(16→8)', 'ReLU3_0', 
        'Linear_1\n(8→4)', 'ReLU3_1', 
        'Linear_2\n(4→2)', 'ReLU3_2'
    ],
    'layer_type': ['Linear', 'Activation', 'Linear', 'Activation', 'Linear', 'Activation'],
    'max_abs_err': [5.16e-08, 2.34e-01, 1.73e-01, 2.53e-01, 7.91e-02, 2.25e-01],
    'mean_abs_err': [1.73e-08, 1.55e-01, 7.76e-02, 1.89e-01, 4.78e-02, 1.95e-01],
    'rms_err': [2.40e-08, 1.69e-01, 9.89e-02, 2.02e-01, 5.71e-02, 1.97e-01],
    'max_rel_err': [1.09e-06, 5.06e+00, 3.05e+00, 5.91e+00, 1.12e+00, 2.33e+00],
    'dims': [8, 8, 4, 4, 2, 2]
}

# All models summary
all_models_summary = {
    'model': ['MNIST_MLP', 'BCW_FC', 'LeNet_FC', 'Audio1D_FC', 'Small_FC'],
    'architecture': ['784→128→32→10', '64→32→16→10', '256→120→84→10', '2000→5', '16→8→4→2'],
    'num_layers': [3, 3, 3, 1, 3],
    'total_rms': [3.765e-01, 4.074e-01, 3.955e-01, 1.362e-01, 3.485e-01],
    'exec_time': [70.86, 20.66, 99.82, 18.54, 8.23]
}


def plot_rms_progression(data, title, filename):
    """Plot RMS error progression through layers."""
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db' if t == 'Linear' else '#e74c3c' for t in df['layer_type']]
    bars = ax.bar(df['layer_idx'], df['rms_err'], color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('RMS Error (log scale)', fontsize=12)
    ax.set_title(f'{title}\nLayer-by-Layer Accumulated RMS Error', fontsize=14)
    ax.set_xticks(df['layer_idx'])
    ax.set_xticklabels(df['layer_name'], rotation=45, ha='right', fontsize=10)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df['rms_err']):
        height = bar.get_height()
        ax.annotate(f'{val:.2e}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Linear Layer'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='ReLU3 Activation')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


def plot_comparison(data1, data2, label1, label2, filename):
    """Compare RMS error between two models."""
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(df1['layer_idx']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1['rms_err'], width, label=label1, color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, df2['rms_err'], width, label=label2, color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('RMS Error (log scale)', fontsize=12)
    ax.set_title('Model Comparison: Layer-by-Layer RMS Error', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['L0\n(Linear)', 'L1\n(ReLU3)', 'L2\n(Linear)', 
                        'L3\n(ReLU3)', 'L4\n(Linear)', 'L5\n(ReLU3)'],
                       fontsize=10)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


def plot_error_components(data, title, filename):
    """Plot max, mean, and RMS error together."""
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(df['layer_idx']))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['max_abs_err'], width, label='Max Abs Error', color='#e74c3c')
    bars2 = ax.bar(x, df['rms_err'], width, label='RMS Error', color='#3498db')
    bars3 = ax.bar(x + width, df['mean_abs_err'], width, label='Mean Abs Error', color='#2ecc71')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Error (log scale)', fontsize=12)
    ax.set_title(f'{title}\nError Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['layer_name'], rotation=45, ha='right', fontsize=10)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


def plot_cumulative_error(data, title, filename):
    """Plot cumulative RMS error through layers."""
    df = pd.DataFrame(data)
    
    # Compute cumulative RMS (sqrt of sum of squared RMS errors)
    cumulative_rms = []
    running_sum = 0
    for rms in df['rms_err']:
        running_sum += rms ** 2
        cumulative_rms.append(np.sqrt(running_sum))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['layer_idx'], cumulative_rms, 'o-', linewidth=2, markersize=8, 
            color='#3498db', label='Cumulative RMS')
    ax.fill_between(df['layer_idx'], 0, cumulative_rms, alpha=0.3, color='#3498db')
    
    # Add markers for layer types
    for i, (idx, ltype) in enumerate(zip(df['layer_idx'], df['layer_type'])):
        color = '#e74c3c' if ltype == 'Activation' else '#3498db'
        ax.scatter(idx, cumulative_rms[i], s=100, color=color, zorder=5)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Cumulative RMS Error', fontsize=12)
    ax.set_title(f'{title}\nCumulative Error Progression', fontsize=14)
    ax.set_xticks(df['layer_idx'])
    ax.set_xticklabels(df['layer_name'], rotation=45, ha='right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add final value annotation
    ax.annotate(f'Total: {cumulative_rms[-1]:.4f}',
                xy=(df['layer_idx'].iloc[-1], cumulative_rms[-1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


def generate_latex_table(data, caption, label):
    """Generate LaTeX table from data."""
    df = pd.DataFrame(data)
    
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{|c|l|c|c|c|c|c|}}
\\hline
\\textbf{{Index}} & \\textbf{{Layer}} & \\textbf{{Max Abs}} & \\textbf{{Mean Abs}} & \\textbf{{RMS}} & \\textbf{{Max Rel}} & \\textbf{{Dims}} \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        layer_name = row['layer_name'].replace('\n', ' ')
        latex += f"{row['layer_idx']} & {layer_name} & {row['max_abs_err']:.2e} & {row['mean_abs_err']:.2e} & {row['rms_err']:.2e} & {row['max_rel_err']:.2e} & {row['dims']} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex


def main():
    """Generate all plots and tables."""
    print("=" * 60)
    print("HE Divergence Analysis - Visualization Generator")
    print("=" * 60)
    
    # Create output directory
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plots for all models
    print("\nGenerating plots...")
    
    all_models = [
        (mnist_mlp_data, 'MNIST MLP (784→128→32→10)', 'mnist_mlp'),
        (bcw_fc_data, 'BCW FC (64→32→16→10)', 'bcw_fc'),
        (lenet_fc_data, 'LeNet FC (256→120→84→10)', 'lenet_fc'),
        (audio1d_fc_data, 'Audio1D FC (2000→5)', 'audio1d_fc'),
        (small_fc_data, 'Small FC (16→8→4→2)', 'small_fc'),
    ]
    
    # 1. Individual RMS progression plots
    for data, title, prefix in all_models:
        plot_rms_progression(
            data, title,
            os.path.join(plots_dir, f'{prefix}_rms_progression.png')
        )
    
    # 2. Model comparison (all 3-layer models)
    plot_all_models_comparison(
        [mnist_mlp_data, bcw_fc_data, lenet_fc_data, small_fc_data],
        ['MNIST MLP', 'BCW FC', 'LeNet FC', 'Small FC'],
        os.path.join(plots_dir, 'all_models_comparison.png')
    )
    
    # 3. Summary bar chart
    plot_summary_comparison(
        all_models_summary,
        os.path.join(plots_dir, 'summary_total_rms.png')
    )
    
    # 4. Cumulative error plots
    for data, title, prefix in all_models:
        plot_cumulative_error(
            data, title,
            os.path.join(plots_dir, f'{prefix}_cumulative_error.png')
        )
    
    # 5. Error components (MNIST as example)
    plot_error_components(
        mnist_mlp_data,
        'MNIST MLP (784→128→32→10)',
        os.path.join(plots_dir, 'mnist_error_components.png')
    )
    
    # Generate LaTeX tables
    print("\n" + "=" * 60)
    print("LaTeX Tables")
    print("=" * 60)
    
    for data, title, prefix in all_models:
        print(f"\n--- {title} Table ---")
        print(generate_latex_table(
            data,
            f'{title} Layer-by-Layer Divergence (HE vs Plaintext)',
            f'tab:{prefix}_divergence'
        ))
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    for data, title, prefix in all_models:
        df = pd.DataFrame(data)
        total_rms = np.sqrt(sum(df['rms_err'] ** 2))
        linear_mask = df['layer_type'] == 'Linear'
        act_mask = df['layer_type'] == 'Activation'
        linear_rms = df[linear_mask]['rms_err'].mean() if linear_mask.any() else 0
        act_rms = df[act_mask]['rms_err'].mean() if act_mask.any() else 0
        
        print(f"\n{title}:")
        print(f"  Total accumulated RMS: {total_rms:.6f}")
        print(f"  Avg Linear layer RMS:  {linear_rms:.2e}")
        print(f"  Avg Activation RMS:    {act_rms:.4f}")
        if linear_rms > 0:
            print(f"  Ratio (Act/Linear):    {act_rms/linear_rms:.2e}x")
    
    print("\n" + "=" * 60)
    print("All plots saved to:", plots_dir)
    print("=" * 60)


def plot_all_models_comparison(datasets, labels, filename):
    """Compare RMS error across all models."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Use first 6 layers for comparison (3-layer networks)
    x = np.arange(6)
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (data, label) in enumerate(zip(datasets, labels)):
        df = pd.DataFrame(data)
        if len(df) >= 6:
            bars = ax.bar(x + i*width - 1.5*width, df['rms_err'][:6], width, 
                         label=label, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('RMS Error (log scale)', fontsize=12)
    ax.set_title('All Models Comparison: Layer-by-Layer RMS Error', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['L0\n(Linear)', 'L1\n(ReLU3)', 'L2\n(Linear)', 
                        'L3\n(ReLU3)', 'L4\n(Linear)', 'L5\n(ReLU3)'],
                       fontsize=10)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_summary_comparison(summary_data, filename):
    """Plot summary bar chart of total RMS error."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = summary_data['model']
    total_rms = summary_data['total_rms']
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(models, total_rms, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Total Accumulated RMS Error', fontsize=12)
    ax.set_title('Total RMS Error Comparison Across All Architectures', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, total_rms):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add architecture labels below model names
    for i, (model, arch) in enumerate(zip(models, summary_data['architecture'])):
        ax.annotate(arch, xy=(i, -0.02), ha='center', va='top', fontsize=9,
                   xycoords=('data', 'axes fraction'), color='gray')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


if __name__ == '__main__':
    main()
