"""Compute robustness metrics from shift evaluation results.

Reads CSV files produced by evaluate_shift.py and computes:
- Robustness drop Δ(δ) for each shift level
- Area Under Degradation Curve (AUDC)
- Worst-case performance
- Comparison tables between DoubleCritic and TripleCritic

Usage:
    python compute_robustness.py \
        --results_dir=./results/ \
        --env_name=hopper-medium-v2
"""

import csv
import glob
import os
from typing import Dict, List

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('results_dir', './results/', 'Directory with shift CSVs.')
flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')
flags.DEFINE_string('output_file', '', 'Output file for comparison table.')


def load_results(filepath: str) -> List[Dict]:
    """Load results from a CSV file."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['shift_level'] = float(row['shift_level'])
            row['raw_return'] = float(row['raw_return'])
            row['episode_length'] = float(row['episode_length'])
            row['robustness_drop'] = float(row['robustness_drop'])
            results.append(row)
    return results


def compute_audc(results: List[Dict], shift_type: str) -> float:
    """Compute Area Under Degradation Curve.

    Integrates |Δ(δ)| over shift levels using trapezoidal rule.
    Lower AUDC = more robust.
    """
    filtered = [r for r in results if r['shift_type'] == shift_type]
    filtered.sort(key=lambda x: x['shift_level'])

    if len(filtered) < 2:
        return 0.0

    levels = [r['shift_level'] for r in filtered]
    drops = [abs(r['robustness_drop']) for r in filtered]

    # Trapezoidal integration
    audc = np.trapz(drops, levels)
    return audc


def compute_worst_case(results: List[Dict], shift_type: str) -> float:
    """Compute worst-case return across all shift levels."""
    filtered = [r for r in results if r['shift_type'] == shift_type]
    if not filtered:
        return float('nan')
    return min(r['raw_return'] for r in filtered)


def compute_robustness_threshold(results: List[Dict], shift_type: str,
                                  threshold_pct: float = 0.5) -> float:
    """Find maximum shift level where performance stays above threshold.

    Args:
        threshold_pct: Fraction of baseline performance to maintain (0.5 = 50%).

    Returns:
        Maximum shift level where performance >= threshold_pct * baseline.
    """
    filtered = [r for r in results if r['shift_type'] == shift_type]
    filtered.sort(key=lambda x: x['shift_level'])

    if not filtered:
        return 0.0

    # Find baseline
    if shift_type == 'gravity':
        baseline = next((r['raw_return'] for r in filtered
                        if r['shift_level'] == 1.0), filtered[0]['raw_return'])
    else:
        baseline = next((r['raw_return'] for r in filtered
                        if r['shift_level'] == 0.0), filtered[0]['raw_return'])

    threshold = threshold_pct * baseline
    max_level = filtered[0]['shift_level']

    for r in filtered:
        if r['raw_return'] >= threshold:
            max_level = r['shift_level']

    return max_level


def main(_):
    # Find all result files for this environment
    pattern = os.path.join(FLAGS.results_dir,
                           f'shift_results_{FLAGS.env_name}_*.csv')
    files = glob.glob(pattern)

    if not files:
        print(f"No result files found matching: {pattern}")
        return

    print(f"\n{'='*70}")
    print(f"ROBUSTNESS ANALYSIS — {FLAGS.env_name}")
    print(f"{'='*70}")

    all_configs = {}
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        # Extract critic config from filename
        parts = filename.replace('.csv', '').split('_')
        # Find the part that looks like "2Q" or "3Q"
        critic_label = [p for p in parts if p.endswith('Q')][0] \
            if any(p.endswith('Q') for p in parts) else 'unknown'

        results = load_results(filepath)
        all_configs[critic_label] = results

    # Print per-config analysis
    for label, results in sorted(all_configs.items()):
        print(f"\n--- {label} ---")

        shift_types = set(r['shift_type'] for r in results)
        for st in sorted(shift_types):
            audc = compute_audc(results, st)
            worst = compute_worst_case(results, st)
            threshold = compute_robustness_threshold(results, st)

            print(f"\n  {st}:")
            print(f"    AUDC (lower=better):        {audc:.4f}")
            print(f"    Worst-case return:           {worst:.2f}")
            print(f"    50% robustness threshold:    {threshold:.2f}")

            print(f"    {'Level':<10} {'Return':<12} {'Δ(δ)':<10}")
            print(f"    {'-'*32}")
            for r in sorted([r for r in results if r['shift_type'] == st],
                           key=lambda x: x['shift_level']):
                print(f"    {r['shift_level']:<10.2f} "
                      f"{r['raw_return']:<12.2f} "
                      f"{r['robustness_drop']:<10.4f}")

    # Print comparison table if multiple configs
    if len(all_configs) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")

        labels = sorted(all_configs.keys())
        shift_types = set()
        for results in all_configs.values():
            shift_types.update(r['shift_type'] for r in results)

        for st in sorted(shift_types):
            print(f"\n  {st}:")
            header = f"    {'Level':<10}"
            for label in labels:
                header += f" {label+' Return':<15} {label+' Δ(δ)':<12}"
            print(header)
            print(f"    {'-'*len(header)}")

            # Get all levels
            all_levels = set()
            for results in all_configs.values():
                for r in results:
                    if r['shift_type'] == st:
                        all_levels.add(r['shift_level'])

            for level in sorted(all_levels):
                row = f"    {level:<10.2f}"
                for label in labels:
                    results = all_configs[label]
                    match = [r for r in results
                            if r['shift_type'] == st
                            and r['shift_level'] == level]
                    if match:
                        row += f" {match[0]['raw_return']:<15.2f}"
                        row += f" {match[0]['robustness_drop']:<12.4f}"
                    else:
                        row += f" {'N/A':<15} {'N/A':<12}"
                print(row)

            # AUDC comparison
            print(f"\n    AUDC: ", end="")
            for label in labels:
                audc = compute_audc(all_configs[label], st)
                print(f"{label}={audc:.4f}  ", end="")
            print()


if __name__ == '__main__':
    app.run(main)
