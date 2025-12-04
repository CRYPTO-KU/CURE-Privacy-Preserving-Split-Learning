#!/usr/bin/env python3
"""
Quick Speedup Table Generator for Chat Display
"""
import pandas as pd
import numpy as np

def load_data(cores):
    """Load benchmark data for specific core count"""
    try:
        return pd.read_csv(f'../bench_results_cores{cores}_logn13.csv')
    except FileNotFoundError:
        return pd.DataFrame()

def calculate_speedups():
    """Calculate speedups for all layers and core counts"""
    core_counts = [1, 2, 4, 8, 16, 32, 40]
    
    # Load baseline (1 core) data
    baseline_df = load_data(1)
    if baseline_df.empty:
        print("❌ No baseline data found")
        return
    
    # Get HE operations only (we care about HE speedups)
    baseline_he = baseline_df[baseline_df['mode'] == 'HE'].copy()
    
    # Create speedup table
    speedup_data = []
    
    for cores in [2, 4, 8, 16, 32, 40]:
        current_df = load_data(cores)
        if current_df.empty:
            continue
            
        current_he = current_df[current_df['mode'] == 'HE'].copy()
        
        # Match layers between baseline and current
        for _, baseline_row in baseline_he.iterrows():
            model = baseline_row['model']
            layer = baseline_row['layer']
            
            # Find corresponding row in current data
            current_row = current_he[(current_he['model'] == model) & 
                                   (current_he['layer'] == layer)]
            
            if not current_row.empty:
                current_row = current_row.iloc[0]
                
                # Calculate speedups (baseline_time / current_time)
                baseline_total = baseline_row['forward_time'] + baseline_row['backward_time']
                current_total = current_row['forward_time'] + current_row['backward_time']
                
                if current_total > 0:
                    speedup = baseline_total / current_total
                    
                    speedup_data.append({
                        'model': model,
                        'layer': layer,
                        'layer_type': layer.split('_')[0],  # Extract layer type
                        'cores': cores,
                        'forward_speedup': baseline_row['forward_time'] / current_row['forward_time'] if current_row['forward_time'] > 0 else 0,
                        'backward_speedup': baseline_row['backward_time'] / current_row['backward_time'] if current_row['backward_time'] > 0 else 0,
                        'total_speedup': speedup,
                        'baseline_time': baseline_total,
                        'current_time': current_total
                    })
    
    return pd.DataFrame(speedup_data)

def create_summary_tables(speedup_df):
    """Create summary tables for chat display"""
    if speedup_df.empty:
        print("❌ No speedup data available")
        return
    
    print("🚀 CURE_lib HE Parallelization Speedup Analysis")
    print("=" * 80)
    
    # 1. Average speedup by layer type and core count
    print("\n📊 TABLE 1: AVERAGE SPEEDUP BY LAYER TYPE")
    print("-" * 60)
    
    layer_speedup = speedup_df.groupby(['layer_type', 'cores'])['total_speedup'].mean().unstack(fill_value=0)
    layer_speedup_formatted = layer_speedup.round(2)
    
    print(f"{'Layer Type':<15} {'2 cores':<8} {'4 cores':<8} {'8 cores':<8} {'16 cores':<9} {'32 cores':<9} {'40 cores':<9}")
    print("-" * 75)
    
    for layer_type in layer_speedup_formatted.index:
        row = f"{layer_type:<15}"
        for cores in [2, 4, 8, 16, 32, 40]:
            if cores in layer_speedup_formatted.columns:
                speedup_val = layer_speedup_formatted.loc[layer_type, cores]
                if speedup_val > 0:
                    row += f" {speedup_val:>7.2f}x"
                else:
                    row += f"    {'N/A':>4}"
            else:
                row += f"    {'N/A':>4}"
        print(row)
    
    # 2. Best performing layers (highest speedup at 40 cores)
    print(f"\n📈 TABLE 2: TOP PERFORMING LAYERS (40 CORES)")
    print("-" * 60)
    
    best_40_cores = speedup_df[speedup_df['cores'] == 40].nlargest(10, 'total_speedup')
    
    print(f"{'Model':<10} {'Layer':<20} {'Speedup':<10} {'Baseline (s)':<12} {'Optimized (s)':<12}")
    print("-" * 70)
    
    for _, row in best_40_cores.iterrows():
        print(f"{row['model']:<10} {row['layer']:<20} {row['total_speedup']:>7.2f}x {row['baseline_time']:>10.3f}s {row['current_time']:>11.3f}s")
    
    # 3. Model-wise performance summary
    print(f"\n🎯 TABLE 3: MODEL-WISE SPEEDUP SUMMARY")
    print("-" * 60)
    
    model_speedup = speedup_df.groupby(['model', 'cores'])['total_speedup'].mean().unstack(fill_value=0)
    
    print(f"{'Model':<10} {'2 cores':<8} {'4 cores':<8} {'8 cores':<8} {'16 cores':<9} {'32 cores':<9} {'40 cores':<9}")
    print("-" * 70)
    
    for model in model_speedup.index:
        row = f"{model:<10}"
        for cores in [2, 4, 8, 16, 32, 40]:
            if cores in model_speedup.columns:
                speedup_val = model_speedup.loc[model, cores]
                if speedup_val > 0:
                    row += f" {speedup_val:>7.2f}x"
                else:
                    row += f"    {'N/A':>4}"
            else:
                row += f"    {'N/A':>4}"
        print(row)
    
    # 4. Efficiency analysis (speedup / cores)
    print(f"\n⚡ TABLE 4: PARALLELIZATION EFFICIENCY (Speedup/Cores)")
    print("-" * 60)
    
    efficiency_df = speedup_df.copy()
    efficiency_df['efficiency'] = efficiency_df['total_speedup'] / efficiency_df['cores']
    
    efficiency_summary = efficiency_df.groupby(['layer_type', 'cores'])['efficiency'].mean().unstack(fill_value=0)
    
    print(f"{'Layer Type':<15} {'2 cores':<8} {'4 cores':<8} {'8 cores':<8} {'16 cores':<9} {'32 cores':<9} {'40 cores':<9}")
    print("-" * 75)
    
    for layer_type in efficiency_summary.index:
        row = f"{layer_type:<15}"
        for cores in [2, 4, 8, 16, 32, 40]:
            if cores in efficiency_summary.columns:
                eff_val = efficiency_summary.loc[layer_type, cores]
                if eff_val > 0:
                    row += f" {eff_val:>7.1%}"
                else:
                    row += f"    {'N/A':>4}"
            else:
                row += f"    {'N/A':>4}"
        print(row)
    
    # 5. Key insights
    print(f"\n🔍 KEY INSIGHTS")
    print("-" * 30)
    
    max_speedup = speedup_df['total_speedup'].max()
    max_speedup_row = speedup_df.loc[speedup_df['total_speedup'].idxmax()]
    
    avg_40_core_speedup = speedup_df[speedup_df['cores'] == 40]['total_speedup'].mean()
    
    print(f"• Maximum speedup achieved: {max_speedup:.2f}x")
    print(f"  └─ {max_speedup_row['model']}/{max_speedup_row['layer']} with {max_speedup_row['cores']} cores")
    print(f"• Average speedup at 40 cores: {avg_40_core_speedup:.2f}x")
    print(f"• Best performing layer type: {efficiency_df.groupby('layer_type')['total_speedup'].mean().idxmax()}")
    
    # Count super-linear speedups
    super_linear = speedup_df[speedup_df['total_speedup'] > speedup_df['cores']]
    print(f"• Super-linear speedups: {len(super_linear)} instances")
    
    return speedup_df

def main():
    speedup_df = calculate_speedups()
    if speedup_df is not None and not speedup_df.empty:
        create_summary_tables(speedup_df)
    else:
        print("❌ Failed to generate speedup analysis")

if __name__ == "__main__":
    main()