#!/usr/bin/env python3
"""
Investigate the sudden speedup jump for BCWFC model at 40 cores
"""
import pandas as pd
import numpy as np

def analyze_bcwfc_jump():
    """Analyze the BCWFC speedup jump from 32 to 40 cores"""
    print("🔍 INVESTIGATING BCWFC MODEL SPEEDUP JUMP")
    print("=" * 60)
    
    # Load data for different core counts
    core_counts = [1, 2, 4, 8, 16, 32, 40]
    bcwfc_data = {}
    
    for cores in core_counts:
        try:
            df = pd.read_csv(f'../bench_results_cores{cores}_logn13.csv')
            bcwfc_df = df[df['model'] == 'bcwfc']
            bcwfc_data[cores] = bcwfc_df
            print(f"✅ Loaded {cores} cores: {len(bcwfc_df)} rows")
        except FileNotFoundError:
            print(f"❌ File not found for {cores} cores")
    
    if not bcwfc_data:
        print("❌ No BCWFC data found")
        return
    
    # Get baseline (1 core) data
    baseline_df = bcwfc_data[1]
    baseline_he = baseline_df[baseline_df['mode'] == 'HE']
    
    print(f"\n📊 BCWFC MODEL LAYER-BY-LAYER ANALYSIS")
    print("-" * 80)
    print(f"{'Layer':<20} {'1 core (s)':<12} {'32 cores (s)':<13} {'40 cores (s)':<13} {'32x Speedup':<12} {'40x Speedup':<12}")
    print("-" * 80)
    
    total_improvements = []
    layer_analysis = []
    
    for _, baseline_row in baseline_he.iterrows():
        layer = baseline_row['layer']
        baseline_time = baseline_row['forward_time'] + baseline_row['backward_time']
        
        # Get 32 core time
        if 32 in bcwfc_data:
            core32_he = bcwfc_data[32][(bcwfc_data[32]['mode'] == 'HE') & 
                                      (bcwfc_data[32]['layer'] == layer)]
            if not core32_he.empty:
                core32_time = core32_he.iloc[0]['forward_time'] + core32_he.iloc[0]['backward_time']
                speedup_32 = baseline_time / core32_time if core32_time > 0 else 0
            else:
                core32_time = 0
                speedup_32 = 0
        else:
            core32_time = 0
            speedup_32 = 0
        
        # Get 40 core time
        if 40 in bcwfc_data:
            core40_he = bcwfc_data[40][(bcwfc_data[40]['mode'] == 'HE') & 
                                      (bcwfc_data[40]['layer'] == layer)]
            if not core40_he.empty:
                core40_time = core40_he.iloc[0]['forward_time'] + core40_he.iloc[0]['backward_time']
                speedup_40 = baseline_time / core40_time if core40_time > 0 else 0
            else:
                core40_time = 0
                speedup_40 = 0
        else:
            core40_time = 0
            speedup_40 = 0
        
        # Calculate improvement from 32 to 40 cores
        if speedup_32 > 0 and speedup_40 > 0:
            improvement_ratio = speedup_40 / speedup_32
            total_improvements.append(improvement_ratio)
            
            layer_analysis.append({
                'layer': layer,
                'baseline_time': baseline_time,
                'core32_time': core32_time,
                'core40_time': core40_time,
                'speedup_32': speedup_32,
                'speedup_40': speedup_40,
                'improvement_ratio': improvement_ratio
            })
            
            print(f"{layer:<20} {baseline_time:>10.3f}s {core32_time:>11.3f}s {core40_time:>11.3f}s {speedup_32:>10.2f}x {speedup_40:>10.2f}x")
    
    print("-" * 80)
    
    # Calculate overall statistics
    if total_improvements:
        avg_improvement = np.mean(total_improvements)
        print(f"\n📈 IMPROVEMENT ANALYSIS (32 → 40 cores)")
        print("-" * 40)
        print(f"Average improvement ratio: {avg_improvement:.2f}x")
        print(f"Range: {min(total_improvements):.2f}x to {max(total_improvements):.2f}x")
        
        # Find layers with biggest improvements
        sorted_layers = sorted(layer_analysis, key=lambda x: x['improvement_ratio'], reverse=True)
        
        print(f"\n🚀 BIGGEST IMPROVEMENTS (32 → 40 cores)")
        print("-" * 50)
        for i, layer_info in enumerate(sorted_layers[:3]):
            improvement = layer_info['improvement_ratio']
            layer_name = layer_info['layer']
            print(f"{i+1}. {layer_name}: {improvement:.2f}x improvement")
            print(f"   32 cores: {layer_info['speedup_32']:.2f}x → 40 cores: {layer_info['speedup_40']:.2f}x")
        
        # Check for potential reasons
        print(f"\n🔍 POTENTIAL CAUSES")
        print("-" * 30)
        
        # Check if any layers show super-linear scaling
        super_linear_40 = [x for x in layer_analysis if x['speedup_40'] > 40]
        if super_linear_40:
            print(f"• {len(super_linear_40)} layers show super-linear scaling at 40 cores")
            for layer_info in super_linear_40:
                print(f"  └─ {layer_info['layer']}: {layer_info['speedup_40']:.1f}x speedup")
        
        # Check for memory/cache effects
        significant_improvements = [x for x in layer_analysis if x['improvement_ratio'] > 2.0]
        if significant_improvements:
            print(f"• {len(significant_improvements)} layers show >2x improvement from 32→40 cores")
            print(f"  └─ Possible memory bandwidth/cache optimization at 40 cores")
        
        # Check timing consistency
        very_fast_layers = [x for x in layer_analysis if x['core40_time'] < 0.1]
        if very_fast_layers:
            print(f"• {len(very_fast_layers)} layers have <0.1s execution time at 40 cores")
            print(f"  └─ May indicate measurement noise or very efficient parallelization")
        
        # Compare with hardware characteristics
        print(f"\n💻 HARDWARE CONSIDERATIONS")
        print("-" * 30)
        print(f"• 32 cores: Standard high-performance configuration")
        print(f"• 40 cores: May hit different memory/cache hierarchy")
        print(f"• Improvement ratio {avg_improvement:.2f}x suggests:")
        if avg_improvement > 2.0:
            print(f"  └─ Significant hardware optimization at 40 cores")
            print(f"  └─ Possible memory bandwidth saturation relief")
        elif avg_improvement > 1.5:
            print(f"  └─ Good scaling with additional cores")
        else:
            print(f"  └─ Moderate improvement, as expected")

def compare_all_models_40_core_jump():
    """Compare the 32→40 core jump across all models"""
    print(f"\n🔍 COMPARING 32→40 CORE JUMP ACROSS ALL MODELS")
    print("=" * 60)
    
    models = ['mnistfc', 'lenet', 'bcwfc', 'audio1d']
    model_improvements = {}
    
    for model in models:
        try:
            # Load 32 and 40 core data
            df_32 = pd.read_csv(f'../bench_results_cores32_logn13.csv')
            df_40 = pd.read_csv(f'../bench_results_cores40_logn13.csv')
            df_1 = pd.read_csv(f'../bench_results_cores1_logn13.csv')
            
            # Filter for this model and HE mode
            model_32 = df_32[(df_32['model'] == model) & (df_32['mode'] == 'HE')]
            model_40 = df_40[(df_40['model'] == model) & (df_40['mode'] == 'HE')]
            model_1 = df_1[(df_1['model'] == model) & (df_1['mode'] == 'HE')]
            
            if model_32.empty or model_40.empty or model_1.empty:
                continue
            
            # Calculate average speedups
            total_time_1 = (model_1['forward_time'] + model_1['backward_time']).sum()
            total_time_32 = (model_32['forward_time'] + model_32['backward_time']).sum()
            total_time_40 = (model_40['forward_time'] + model_40['backward_time']).sum()
            
            speedup_32 = total_time_1 / total_time_32 if total_time_32 > 0 else 0
            speedup_40 = total_time_1 / total_time_40 if total_time_40 > 0 else 0
            
            improvement = speedup_40 / speedup_32 if speedup_32 > 0 else 0
            
            model_improvements[model] = {
                'speedup_32': speedup_32,
                'speedup_40': speedup_40,
                'improvement': improvement
            }
            
        except FileNotFoundError:
            print(f"❌ Data not found for {model}")
    
    print(f"{'Model':<10} {'32 cores':<10} {'40 cores':<10} {'Improvement':<12}")
    print("-" * 45)
    
    for model, data in model_improvements.items():
        print(f"{model:<10} {data['speedup_32']:>8.2f}x {data['speedup_40']:>8.2f}x {data['improvement']:>10.2f}x")
    
    # Find the biggest improver
    if model_improvements:
        biggest_improver = max(model_improvements.items(), key=lambda x: x[1]['improvement'])
        print(f"\n🏆 Biggest improvement: {biggest_improver[0]} ({biggest_improver[1]['improvement']:.2f}x)")

def main():
    analyze_bcwfc_jump()
    compare_all_models_40_core_jump()

if __name__ == "__main__":
    main()