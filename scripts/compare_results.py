#!/usr/bin/env python3
"""
CURE Artifact - Results Comparison Script
PoPETs 2026 Artifact Evaluation

Compares benchmark results with reference values from the paper.
Reports whether results are within acceptable tolerance (default: 5%).

Usage:
    python3 scripts/compare_results.py --new <your_results.csv> --ref <reference.csv>
    python3 scripts/compare_results.py --new bench_results_cores4_logn13.csv --ref results/bench0803/bench_results_cores4_logn13.csv

Authors: Halil Ibrahim Kanpak, Aqsa Shabbir, Esra Genç, Alptekin Küpçü, Sinem Sav
"""

import argparse
import csv
import sys
from pathlib import Path


def load_csv(filepath):
    """Load CSV file and return list of dictionaries."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def parse_numeric(value):
    """Parse numeric value, handling various formats."""
    if value is None or value == '' or value == 'N/A':
        return None
    try:
        # Remove common units (ms, s, etc.)
        cleaned = value.strip().replace('ms', '').replace('s', '').strip()
        return float(cleaned)
    except ValueError:
        return None


def compare_results(new_file, ref_file, tolerance=0.05, key_column='Layer'):
    """
    Compare two CSV files and report differences.
    
    Args:
        new_file: Path to new results CSV
        ref_file: Path to reference results CSV
        tolerance: Acceptable relative difference (default 5%)
        key_column: Column to use as key for matching rows
    
    Returns:
        Tuple of (passed, total, details)
    """
    new_data = load_csv(new_file)
    ref_data = load_csv(ref_file)
    
    if not new_data or not ref_data:
        print("ERROR: One or both CSV files are empty or invalid.")
        return 0, 0, []
    
    # Get numeric columns (exclude key column)
    numeric_cols = []
    for col in new_data[0].keys():
        if col != key_column:
            val = parse_numeric(new_data[0].get(col))
            if val is not None:
                numeric_cols.append(col)
    
    # Create lookup by key
    ref_lookup = {}
    for row in ref_data:
        key = row.get(key_column, '')
        if key:
            ref_lookup[key] = row
    
    results = []
    passed = 0
    total = 0
    
    for new_row in new_data:
        key = new_row.get(key_column, '')
        if key not in ref_lookup:
            continue
        
        ref_row = ref_lookup[key]
        
        for col in numeric_cols:
            new_val = parse_numeric(new_row.get(col))
            ref_val = parse_numeric(ref_row.get(col))
            
            if new_val is None or ref_val is None:
                continue
            
            total += 1
            
            # Calculate relative difference
            if ref_val != 0:
                rel_diff = abs(new_val - ref_val) / abs(ref_val)
            else:
                rel_diff = 0 if new_val == 0 else 1.0
            
            is_pass = rel_diff <= tolerance
            if is_pass:
                passed += 1
            
            results.append({
                'key': key,
                'column': col,
                'new': new_val,
                'ref': ref_val,
                'diff_pct': rel_diff * 100,
                'pass': is_pass
            })
    
    return passed, total, results


def print_report(passed, total, results, tolerance):
    """Print comparison report."""
    print("=" * 70)
    print("CURE Artifact - Results Comparison Report")
    print("=" * 70)
    print(f"Tolerance: {tolerance * 100:.1f}%")
    print(f"Comparisons: {passed}/{total} PASSED")
    print()
    
    if total == 0:
        print("WARNING: No numeric comparisons could be made.")
        print("Check that both CSV files have matching columns and row keys.")
        return
    
    pass_rate = (passed / total) * 100
    
    # Summary
    if pass_rate >= 95:
        print(f"✓ RESULT: PASS ({pass_rate:.1f}% within tolerance)")
        print("  Results are reproducible within expected tolerance.")
    elif pass_rate >= 80:
        print(f"~ RESULT: PARTIAL ({pass_rate:.1f}% within tolerance)")
        print("  Most results match. Differences may be due to hardware variations.")
    else:
        print(f"✗ RESULT: CHECK NEEDED ({pass_rate:.1f}% within tolerance)")
        print("  Significant differences detected. Check hardware/configuration.")
    
    print()
    print("-" * 70)
    print("Detailed Comparison:")
    print("-" * 70)
    print(f"{'Row':<20} {'Column':<15} {'New':>12} {'Ref':>12} {'Diff%':>8} {'Status':<6}")
    print("-" * 70)
    
    for r in results:
        status = "PASS" if r['pass'] else "FAIL"
        print(f"{r['key']:<20} {r['column']:<15} {r['new']:>12.2f} {r['ref']:>12.2f} {r['diff_pct']:>7.1f}% {status:<6}")
    
    print("-" * 70)
    print()
    
    # Show failures
    failures = [r for r in results if not r['pass']]
    if failures:
        print(f"Failures ({len(failures)}):")
        for r in failures:
            print(f"  - {r['key']}/{r['column']}: {r['new']:.2f} vs {r['ref']:.2f} ({r['diff_pct']:.1f}% diff)")


def main():
    parser = argparse.ArgumentParser(
        description='Compare CURE benchmark results with reference values.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 compare_results.py --new results.csv --ref reference.csv
  python3 compare_results.py --new results.csv --ref reference.csv --tolerance 0.10
  python3 compare_results.py --new results.csv --ref reference.csv --key Model
        """
    )
    parser.add_argument('--new', required=True, help='Path to new results CSV')
    parser.add_argument('--ref', required=True, help='Path to reference results CSV')
    parser.add_argument('--tolerance', type=float, default=0.05, 
                        help='Acceptable relative difference (default: 0.05 = 5%%)')
    parser.add_argument('--key', default='Layer',
                        help='Column name to use as row key (default: Layer)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.new).exists():
        print(f"ERROR: New results file not found: {args.new}")
        sys.exit(1)
    if not Path(args.ref).exists():
        print(f"ERROR: Reference file not found: {args.ref}")
        sys.exit(1)
    
    print(f"Comparing: {args.new}")
    print(f"Reference: {args.ref}")
    print()
    
    passed, total, results = compare_results(
        args.new, args.ref, 
        tolerance=args.tolerance,
        key_column=args.key
    )
    
    print_report(passed, total, results, args.tolerance)
    
    # Exit code: 0 if >80% pass, 1 otherwise
    pass_rate = (passed / total * 100) if total > 0 else 0
    sys.exit(0 if pass_rate >= 80 else 1)


if __name__ == '__main__':
    main()
