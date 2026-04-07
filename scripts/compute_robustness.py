# -*- coding: utf-8 -*-
"""Compute robustness metrics from shift evaluation results.

Usage:
    python scripts/compute_robustness.py \
        --results_dir=./results/ \
        --env_name=hopper-medium-v2
"""

import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('results_dir', './results/', 'Directory with shift CSVs.')
flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')


def load_results(filepath):
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


def compute_audc(results, shift_type):
    filtered = sorted([r for r in results if r['shift_type'] == shift_type],
                      key=lambda x: x['shift_level'])
    if len(filtered) < 2:
        return 0.0
    levels = [r['shift_level'] for r in filtered]
    drops = [abs(r['robustness_drop']) for r in filtered]
    return np.trapz(drops, levels)


def main(_):
    pattern = os.path.join(FLAGS.results_dir,
                           f'shift_{FLAGS.env_name}_*.csv')
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
        parts = filename.replace('.csv', '').split('_')
        critic_label = [p for p in parts if p.endswith('Q')][0] \
            if any(p.endswith('Q') for p in parts) else 'unknown'
        # Extract tau from filename if present (e.g., shift_..._2Q_seed42_tau0.5.csv)
        tau_parts = [p for p in parts if p.startswith('tau')]
        tau_label = f'_{tau_parts[0]}' if tau_parts else ''
        config_key = f'{critic_label}{tau_label}'
        all_configs[config_key] = load_results(filepath)

    for label, results in sorted(all_configs.items()):
        print(f"\n--- {label} ---")
        shift_types = set(r['shift_type'] for r in results)
        for st in sorted(shift_types):
            audc = compute_audc(results, st)
            worst = min(r['raw_return'] for r in results if r['shift_type'] == st)
            print(f"\n  {st}: AUDC={audc:.4f}, Worst-case={worst:.2f}")
            for r in sorted([r for r in results if r['shift_type'] == st],
                           key=lambda x: x['shift_level']):
                print(f"    {r['shift_level']:<8.2f} return={r['raw_return']:<10.2f} "
                      f"drop={r['robustness_drop']:<8.4f}")

    if len(all_configs) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        labels = sorted(all_configs.keys())
        shift_types = set()
        for results in all_configs.values():
            shift_types.update(r['shift_type'] for r in results)
        for st in sorted(shift_types):
            print(f"\n  {st} AUDC: ", end="")
            for label in labels:
                audc = compute_audc(all_configs[label], st)
                print(f"{label}={audc:.4f}  ", end="")
            print()

    # ── Write summary CSV ──
    summary_file = os.path.join(FLAGS.results_dir,
                                f'summary_{FLAGS.env_name}.csv')
    summary_rows = []
    for label, results in sorted(all_configs.items()):
        shift_types = sorted(set(r['shift_type'] for r in results))
        for st in shift_types:
            audc = compute_audc(results, st)
            filtered = [r for r in results if r['shift_type'] == st]
            worst = min(r['raw_return'] for r in filtered)
            baseline_levels = {'gravity': 1.0, 'obs_noise': 0.0,
                               'friction': 1.0, 'reward_perturb': 0.0}
            bl = baseline_levels.get(st, 0)
            baseline_return = next(
                (r['raw_return'] for r in filtered if r['shift_level'] == bl), 0)
            summary_rows.append({
                'env': FLAGS.env_name,
                'config': label,
                'shift_type': st,
                'baseline_return': f'{baseline_return:.2f}',
                'worst_return': f'{worst:.2f}',
                'audc': f'{audc:.4f}',
            })

    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'env', 'config', 'shift_type',
            'baseline_return', 'worst_return', 'audc'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary written to {summary_file}")


if __name__ == '__main__':
    app.run(main)
