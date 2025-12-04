#!/usr/bin/env python3
"""
Comprehensive Cut Layer Aggregation Analysis for CURE_lib
Creates comparison tables and visualizations for different core counts
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Enhanced blueish color palette for different core counts
CORE_COLORS = {
    1: '#bbdefb',    # Very light blue
    2: '#90caf9',    # Light blue  
    4: '#64b5f6',    # Medium light blue
    8: '#42a5f5',    # Medium blue
    16: '#2196f3',   # Blue
    32: '#1976d2',   # Dark blue
    40: '#0d47a1'    # Very dark blue
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_all_data():
    """Load all benchmark data across different core counts"""
    core_counts = [1, 2, 4, 8, 16, 32, 40]
    all_data = {}
    
    for cores in core_counts:
        file_path = Path(f'../bench_results_cores{cores}_logn13.csv')
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_data[cores] = df
            print(f"✅ Loaded data for {cores} cores: {len(df)} rows")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    return all_data

def process_model_data(df, model_name):
    """Process data for a specific model and create cut aggregations"""
    # Filter for the specific model
    model_data = df[df['model'] == model_name].copy()
    if model_data.empty:
        return None
    
    # Group every two rows (HE + Plain pair)
    groups = []
    he_rows = model_data[model_data['mode'] == 'HE']
    plain_rows = model_data[model_data['mode'] == 'Plain']
    
    # Match HE and Plain rows by layer name
    for _, he_row in he_rows.iterrows():
        layer_name = he_row['layer']
        plain_row = plain_rows[plain_rows['layer'] == layer_name]
        if not plain_row.empty:
            groups.append({
                'layer': layer_name,
                'he_forward': he_row['forward_time'],
                'he_backward': he_row['backward_time'],
                'he_update': he_row['update_time'],
                'plain_forward': plain_row.iloc[0]['forward_time'],
                'plain_backward': plain_row.iloc[0]['backward_time'],
                'plain_update': plain_row.iloc[0]['update_time']
            })
    
    return groups

def compute_cut_aggregations(layer_groups):
    """Compute aggregated times for each possible cut position"""
    if not layer_groups:
        return []
    
    # Build cut labels
    layer_names = [group['layer'] for group in layer_groups]
    cut_labels = ['All Client'] + layer_names
    
    records = []
    for cut in range(len(layer_groups) + 1):
        forward_total = 0.0
        backward_total = 0.0
        update_total = 0.0
        
        for i, group in enumerate(layer_groups):
            if i < cut:
                # Use HE timings (server side)
                forward_total += group['he_forward']
                backward_total += group['he_backward'] 
                update_total += group['he_update']
            else:
                # Use Plain timings (client side)
                forward_total += group['plain_forward']
                backward_total += group['plain_backward']
                update_total += group['plain_update']
        
        total_time = forward_total + backward_total + update_total
        
        records.append({
            'cut_position': cut,
            'cut_label': cut_labels[cut],
            'forward_time': forward_total,
            'backward_time': backward_total, 
            'update_time': update_total,
            'total_time': total_time
        })
    
    return records

def convert_to_hours_minutes(seconds):
    """Convert seconds to hours/minutes for better readability"""
    # Multiply by 750 as requested
    scaled_seconds = seconds * 750
    
    if scaled_seconds >= 3600:  # >= 1 hour
        hours = scaled_seconds / 3600
        return hours, 'hours'
    else:  # < 1 hour, use minutes
        minutes = scaled_seconds / 60
        return minutes, 'minutes'

def create_cut_comparison_visualization(all_results, output_dir):
    """Create comprehensive visualization comparing cuts across core counts"""
    models = list(all_results.keys())
    
    for model in models:
        if not all_results[model]:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{model.upper()} Model: Cut Position Analysis Across Core Counts', 
                     fontsize=18, fontweight='bold')
        
        # Collect data for all core counts
        core_counts = sorted(all_results[model].keys())
        
        # Plot 1: Total time by cut position (connected scatter)
        ax1 = axes[0, 0]
        for cores in core_counts:
            data = all_results[model][cores]
            if data:
                df = pd.DataFrame(data)
                # Convert to hours/minutes
                scaled_times = []
                time_unit = None
                for total_time in df['total_time']:
                    scaled_val, unit = convert_to_hours_minutes(total_time)
                    scaled_times.append(scaled_val)
                    time_unit = unit
                
                ax1.plot(range(len(df)), scaled_times, 
                        marker='o', linewidth=2, markersize=6,
                        color=CORE_COLORS[cores], label=f'{cores} cores')
        
        ax1.set_xlabel('Cut Position')
        ax1.set_ylabel(f'Total Time ({time_unit if "time_unit" in locals() else "minutes"})')
        ax1.set_title('Total Time vs Cut Position')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Forward vs Backward time breakdown
        ax2 = axes[0, 1]
        # Use 32 cores as representative
        if 32 in all_results[model] and all_results[model][32]:
            data_32 = pd.DataFrame(all_results[model][32])
            cut_positions = range(len(data_32))
            
            forward_scaled = [convert_to_hours_minutes(t)[0] for t in data_32['forward_time']]
            backward_scaled = [convert_to_hours_minutes(t)[0] for t in data_32['backward_time']]
            
            ax2.plot(cut_positions, forward_scaled, marker='o', linewidth=2,
                    color=CORE_COLORS[32], label='Forward', alpha=0.8)
            ax2.plot(cut_positions, backward_scaled, marker='s', linewidth=2,
                    color=CORE_COLORS[16], label='Backward', alpha=0.8)
            
            ax2.set_xlabel('Cut Position')
            ax2.set_ylabel(f'Time ({time_unit if "time_unit" in locals() else "minutes"})')
            ax2.set_title('Forward vs Backward Time (32 cores)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Speedup analysis (1 core vs others)
        ax3 = axes[1, 0]
        if 1 in all_results[model] and all_results[model][1]:
            baseline_data = pd.DataFrame(all_results[model][1])
            baseline_times = baseline_data['total_time'].values
            
            for cores in [8, 16, 32, 40]:
                if cores in all_results[model] and all_results[model][cores]:
                    parallel_data = pd.DataFrame(all_results[model][cores])
                    parallel_times = parallel_data['total_time'].values
                    
                    # Calculate speedup
                    speedups = []
                    for i in range(min(len(baseline_times), len(parallel_times))):
                        if parallel_times[i] > 0:
                            speedup = baseline_times[i] / parallel_times[i]
                            speedups.append(speedup)
                        else:
                            speedups.append(0)
                    
                    ax3.plot(range(len(speedups)), speedups, marker='o', linewidth=2,
                            color=CORE_COLORS[cores], label=f'{cores} cores')
            
            ax3.set_xlabel('Cut Position')
            ax3.set_ylabel('Speedup vs 1 core')
            ax3.set_title('Parallelization Speedup by Cut Position')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Optimal cut position by core count
        ax4 = axes[1, 1]
        optimal_cuts = []
        core_list = []
        
        for cores in core_counts:
            if cores in all_results[model] and all_results[model][cores]:
                data = pd.DataFrame(all_results[model][cores])
                optimal_cut = data['total_time'].idxmin()
                optimal_cuts.append(optimal_cut)
                core_list.append(cores)
        
        if optimal_cuts:
            ax4.plot(core_list, optimal_cuts, marker='o', linewidth=3, markersize=8,
                    color=CORE_COLORS[32], alpha=0.8)
            ax4.set_xlabel('Number of Cores')
            ax4.set_ylabel('Optimal Cut Position')
            ax4.set_title('Optimal Cut Position vs Core Count')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model}_cut_analysis_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_tables(all_results, output_dir):
    """Create detailed summary tables for each model"""
    for model in all_results:
        if not all_results[model]:
            continue
            
        print(f"\n{'='*80}")
        print(f"{model.upper()} MODEL - CUT POSITION ANALYSIS")
        print(f"{'='*80}")
        
        # Create comparison table
        core_counts = sorted(all_results[model].keys())
        
        if not core_counts:
            continue
            
        # Get cut labels from first available dataset
        sample_data = None
        for cores in core_counts:
            if all_results[model][cores]:
                sample_data = all_results[model][cores]
                break
        
        if not sample_data:
            continue
            
        cut_labels = [item['cut_label'] for item in sample_data]
        
        # Build comparison dataframe
        comparison_data = []
        for i, cut_label in enumerate(cut_labels):
            row = {'Cut Position': cut_label}
            
            for cores in core_counts:
                if cores in all_results[model] and all_results[model][cores]:
                    data = all_results[model][cores]
                    if i < len(data):
                        total_time = data[i]['total_time']
                        scaled_time, unit = convert_to_hours_minutes(total_time)
                        row[f'{cores} cores'] = f'{scaled_time:.2f} {unit}'
                    else:
                        row[f'{cores} cores'] = 'N/A'
                else:
                    row[f'{cores} cores'] = 'N/A'
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        comparison_df.to_csv(output_dir / f'{model}_cut_comparison.csv', index=False)
        
        # Print formatted table
        print(comparison_df.to_string(index=False))
        
        # Find optimal cuts
        print(f"\n📊 OPTIMAL CUT POSITIONS FOR {model.upper()}:")
        print("-" * 60)
        
        for cores in core_counts:
            if cores in all_results[model] and all_results[model][cores]:
                data = all_results[model][cores]
                if data:
                    min_idx = min(range(len(data)), key=lambda i: data[i]['total_time'])
                    optimal_cut = data[min_idx]
                    total_time = optimal_cut['total_time']
                    scaled_time, unit = convert_to_hours_minutes(total_time)
                    
                    print(f"{cores:>2} cores: {optimal_cut['cut_label']:<20} "
                          f"(Total: {scaled_time:.2f} {unit})")

def main():
    print("🚀 Starting Cut Layer Aggregation Analysis")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('.')
    output_dir.mkdir(exist_ok=True)
    
    # Load all data
    all_data = load_all_data()
    if not all_data:
        print("❌ No data loaded. Check file paths.")
        return
    
    # Process each model across all core counts
    all_results = {}
    models = ['mnistfc', 'lenet', 'bcwfc', 'audio1d']
    
    for model in models:
        print(f"\n📊 Processing {model.upper()} model...")
        all_results[model] = {}
        
        for cores, df in all_data.items():
            layer_groups = process_model_data(df, model)
            if layer_groups:
                cut_results = compute_cut_aggregations(layer_groups)
                all_results[model][cores] = cut_results
                print(f"  ✅ {cores} cores: {len(cut_results)} cut positions")
            else:
                print(f"  ⚠️  {cores} cores: No data found")
    
    # Create visualizations
    print(f"\n🎨 Creating comprehensive visualizations...")
    create_cut_comparison_visualization(all_results, output_dir)
    
    # Create summary tables
    print(f"\n📋 Generating summary tables...")
    create_summary_tables(all_results, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}")
    print(f"\nGenerated files:")
    for model in models:
        print(f"  📊 {model}_cut_analysis_comprehensive.png")
        print(f"  📋 {model}_cut_comparison.csv")

if __name__ == "__main__":
    main()