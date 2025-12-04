#!/usr/bin/env python3
"""
Simple Cut Analysis Demo Script
Following the exact format requested by the user
"""
import pandas as pd
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_model_data(model_name, cores=1):
    """Load data for a specific model and core count"""
    file_path = f'../bench_results_cores{cores}_logn13.csv'
    try:
        df = pd.read_csv(file_path)
        return df[df['model'] == model_name]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

def create_cut_analysis_table(model_name, cores=1):
    """Create cut analysis table exactly as requested"""
    print(f"\n{'='*80}")
    print(f"CUT ANALYSIS FOR {model_name.upper()} MODEL ({cores} cores)")
    print(f"{'='*80}")
    
    # Load the raw CSV data
    df = load_model_data(model_name, cores)
    if df.empty:
        print("No data available")
        return None
    
    # Group every two rows into a single "layer occurrence" (HE row + Plain row)
    groups = []
    he_rows = df[df['mode'] == 'HE']
    plain_rows = df[df['mode'] == 'Plain']
    
    for _, he_row in he_rows.iterrows():
        layer_name = he_row['layer']
        plain_row = plain_rows[plain_rows['layer'] == layer_name]
        if not plain_row.empty:
            groups.append({
                'layer': layer_name,
                'he_data': he_row,
                'plain_data': plain_row.iloc[0]
            })
    
    # Build cut labels: "All client" + each layer name in sequence
    layers = [group['layer'] for group in groups]
    cut_labels = ['All client'] + layers
    
    # For each cut position, sum HE times for layers before the cut and Plain times after
    records = []
    for cut in range(len(groups) + 1):
        fwd = 0.0
        bwd = 0.0
        upd = 0.0
        
        for i, grp in enumerate(groups):
            if i < cut:
                # Server side (HE)
                fwd += grp['he_data']['forward_time']
                bwd += grp['he_data']['backward_time']
                upd += grp['he_data']['update_time']
            else:
                # Client side (Plain)
                fwd += grp['plain_data']['forward_time']
                bwd += grp['plain_data']['backward_time']
                upd += grp['plain_data']['update_time']
        
        # Apply scaling factor (×750) and convert to hours/minutes
        total_seconds = (fwd + bwd + upd) * 750
        
        if total_seconds >= 3600:  # >= 1 hour
            total_time_str = f"{total_seconds/3600:.2f} hours"
        else:
            total_time_str = f"{total_seconds/60:.2f} minutes"
        
        records.append({
            'Cut': cut_labels[cut],
            'Forward (s)': fwd,
            'Backward (s)': bwd,
            'Update (s)': upd,
            'Total (scaled)': total_time_str,
            'Total (raw)': fwd + bwd + upd
        })
    
    result_df = pd.DataFrame(records)
    
    # Display the final table
    print(result_df[['Cut', 'Forward (s)', 'Backward (s)', 'Update (s)', 'Total (scaled)']].to_string(index=False))
    
    return result_df

def create_comparison_visualization():
    """Create visualization comparing different core counts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cut Position Analysis: Core Count Comparison', fontsize=16, fontweight='bold')
    
    models = ['lenet', 'mnistfc', 'bcwfc', 'audio1d']
    core_counts = [1, 2, 4, 8, 16, 32, 40]
    
    # Color scheme: dark blue for 40 cores, light blue for 1 core
    colors = {
        1: '#bbdefb',    # Light blue
        2: '#90caf9',
        4: '#64b5f6',
        8: '#42a5f5',
        16: '#2196f3',
        32: '#1976d2',
        40: '#0d47a1'    # Dark blue
    }
    
    for i, model in enumerate(models):
        ax = axes[i//2, i%2]
        
        for cores in core_counts:
            # Get cut analysis data
            df = load_model_data(model, cores)
            if df.empty:
                continue
                
            # Process groups
            groups = []
            he_rows = df[df['mode'] == 'HE']
            plain_rows = df[df['mode'] == 'Plain']
            
            for _, he_row in he_rows.iterrows():
                layer_name = he_row['layer']
                plain_row = plain_rows[plain_rows['layer'] == layer_name]
                if not plain_row.empty:
                    groups.append({
                        'layer': layer_name,
                        'he_data': he_row,
                        'plain_data': plain_row.iloc[0]
                    })
            
            # Calculate total times for each cut
            cut_times = []
            for cut in range(len(groups) + 1):
                total_time = 0.0
                for j, grp in enumerate(groups):
                    if j < cut:
                        # Server side (HE)
                        total_time += grp['he_data']['forward_time'] + grp['he_data']['backward_time'] + grp['he_data']['update_time']
                    else:
                        # Client side (Plain)
                        total_time += grp['plain_data']['forward_time'] + grp['plain_data']['backward_time'] + grp['plain_data']['update_time']
                
                # Apply scaling (×750) and convert to minutes
                scaled_time = (total_time * 750) / 60  # Convert to minutes
                cut_times.append(scaled_time)
            
            # Plot with connected lines
            x_positions = list(range(len(cut_times)))
            ax.plot(x_positions, cut_times, marker='o', linewidth=2, markersize=6,
                   color=colors[cores], label=f'{cores} cores', alpha=0.8)
        
        ax.set_title(f'{model.upper()} Model')
        ax.set_xlabel('Cut Position')
        ax.set_ylabel('Total Time (minutes)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('cut_comparison_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Visualization saved as 'cut_comparison_visualization.png'")

def main():
    """Main function demonstrating the cut analysis"""
    print("🚀 Cut Layer Analysis Demo")
    print("Following the exact format requested")
    
    # Example: Analyze LENET model with 1 core
    lenet_table = create_cut_analysis_table('lenet', cores=1)
    
    # Example: Analyze MNISTFC model with 32 cores
    mnistfc_table = create_cut_analysis_table('mnistfc', cores=32)
    
    # Create comparison visualization
    print(f"\n🎨 Creating comparison visualization...")
    create_comparison_visualization()
    
    print(f"\n✅ Demo complete!")
    print(f"\nThis analysis shows:")
    print(f"  • Total times are scaled by ×750 and converted to hours/minutes")
    print(f"  • Cut positions from 'All client' to each layer")
    print(f"  • Comparison across different core counts")
    print(f"  • Connected scatter plots with blue color scheme")

if __name__ == "__main__":
    main()