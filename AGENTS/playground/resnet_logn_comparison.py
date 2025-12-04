#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

# Blue color palette
BLUE_COLORS = ['#1f77b4', '#4292c6', '#6baed6', '#9ecae1']

# Layer labels for ResNet (for x-axis)
LAYER_LABELS = {
    0: 'All Plaintext',
    1: 'Conv2D 3→64 (7×7)',
    2: 'ReLU3',
    3: 'Conv2D 64→64 (3×3)',
    4: 'ReLU3',
    5: 'Conv2D 64→64 (3×3)',
    6: 'ReLU3'
}

def plot_resnet_logn_comparison(csv_path, output_path, cores='40'):
    df = pd.read_csv(csv_path, dtype={"logN": str, "num_cores": str})
    model_data = df[(df['model'] == 'resnet') & (df['num_cores'] == str(cores))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, logn in enumerate(['13', '14', '15', '16']):
        data = model_data[model_data['logN'] == logn].sort_values('cut_position')
        if data.empty:
            continue
        total_time = data['forward_time_total'] + data['backprop_time_total']
        x_labels = [LAYER_LABELS.get(cut, f'Cut {cut}') for cut in data['cut_position']]
        ax.plot(range(len(data)), total_time, 'o-', color=BLUE_COLORS[i], linewidth=2, markersize=8, label=f'logN {logn}')
    
    ax.set_xlabel('Cut Position', fontsize=12)
    ax.set_ylabel('Total Training Time per Sample (s)', fontsize=12)
    ax.set_title(f'ResNet: Training Time per Sample vs Cut Position (cores={cores})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    # Use corrected aggregates by default
    csv_path = 'AGENTS/results/cut_aggregates_corrected.csv'
    output_path = 'AGENTS/playground/resnet_logn_comparison.png'
    plot_resnet_logn_comparison(csv_path, output_path) 