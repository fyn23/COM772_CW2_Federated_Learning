"""
Analyze effect of number of rounds on model performance.
Generates plots of accuracy and loss vs rounds for a fixed client count (auto-detected or provided).

Usage:
    python data_analysis/rounds_scaling.py --clients 20

If --clients is not provided, the script will pick the most common client count present in the filenames.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import argparse

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "round_tuning"
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# CLI
parser = argparse.ArgumentParser(description="Rounds scaling analysis")
parser.add_argument("--clients", type=int, default=None, help="Client count to filter files by (default: auto-pick most common)")
parser.add_argument("--pattern", type=str, default="*rounds*.csv", help="Glob pattern to find result CSVs")
args = parser.parse_args()

# Find files
csv_files = list(RESULTS_DIR.glob(args.pattern))
if not csv_files:
    print(f"No CSV files found in {RESULTS_DIR!s} with pattern {args.pattern}")
    raise SystemExit(1)

# Parse files to extract (clients, rounds)
records = []
for f in csv_files:
    name = f.name
    m_clients = re.search(r'(\d+)clients', name)
    m_rounds = re.search(r'_(\d+)rounds_', name)
    if not m_clients or not m_rounds:
        # skip unexpected names
        continue
    clients = int(m_clients.group(1))
    rounds = int(m_rounds.group(1))
    records.append({'path': f, 'clients': clients, 'rounds': rounds})

if not records:
    print("No matching files with clients/rounds in filename.")
    raise SystemExit(1)

df_records = pd.DataFrame(records)

# Decide client count to use
if args.clients is None:
    # pick most common
    clients_to_use = int(df_records['clients'].mode().iloc[0])
    print(f"Auto-selected client count: {clients_to_use}")
else:
    clients_to_use = args.clients
    print(f"Filtering for client count: {clients_to_use}")

# Filter records
use_records = df_records[df_records['clients'] == clients_to_use]
if use_records.empty:
    print(f"No files found for {clients_to_use} clients")
    raise SystemExit(1)

# For each file, extract final metrics at its final round (or use filename rounds)
collected = []
for _, row in use_records.iterrows():
    p = row['path']
    rounds_val = row['rounds']
    df = pd.read_csv(p)
    # Determine the round to read: prefer rounds_val from filename if present in df
    if rounds_val in df['round'].unique():
        target = rounds_val
    else:
        target = int(df['round'].max())
    train_row = df[(df['round'] == target) & (df['phase'] == 'train')]
    eval_row = df[(df['round'] == target) & (df['phase'] == 'evaluate')]
    if train_row.empty or eval_row.empty:
        # fallback to last train/eval rows
        tr = df[df['phase'] == 'train'].iloc[-1]
        er = df[df['phase'] == 'evaluate'].iloc[-1]
        used_round = int(tr['round'])
    else:
        tr = train_row.iloc[0]
        er = eval_row.iloc[0]
        used_round = target
    collected.append({
        'rounds': used_round,
        'train_accuracy': tr['accuracy'],
        'eval_accuracy': er['accuracy'],
        'train_loss': tr['loss'],
        'eval_loss': er['loss'],
        'path': p.name
    })

# Build DataFrame and sort by rounds
col_df = pd.DataFrame(collected).sort_values('rounds')
if col_df.empty:
    print('No data extracted')
    raise SystemExit(1)

print('\nCollected:')
print(col_df[['rounds','train_accuracy','eval_accuracy','train_loss','eval_loss']].to_string(index=False))

# Create combined two-panel plot (Accuracy and Loss)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Rounds Scaling Analysis (clients={clients_to_use})', fontsize=16, fontweight='bold')

# Accuracy subplot
ax1.plot(col_df['rounds'], col_df['train_accuracy'], 'o-', label='Train Accuracy', color='#e74c3c', linewidth=2, markersize=6)
ax1.plot(col_df['rounds'], col_df['eval_accuracy'], 's-', label='Eval Accuracy', color='#3498db', linewidth=2, markersize=6)
ax1.set_xlabel('Rounds', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy vs Rounds')
ax1.grid(alpha=0.3)
ax1.legend()
ax1.set_ylim(0, max(col_df[['train_accuracy', 'eval_accuracy']].max()) * 1.1)

# Loss subplot
ax2.plot(col_df['rounds'], col_df['train_loss'], 'o-', label='Train Loss', color='#e67e22', linewidth=2, markersize=6)
ax2.plot(col_df['rounds'], col_df['eval_loss'], 's-', label='Eval Loss', color='#9b59b6', linewidth=2, markersize=6)
ax2.set_xlabel('Rounds', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Loss vs Rounds')
ax2.grid(alpha=0.3)
ax2.legend()
try:
    ax2.set_ylim(0, max(col_df[['train_loss', 'eval_loss']].max()) * 1.1)
except Exception:
    pass

plt.tight_layout()
output_file = OUTPUT_DIR / f'rounds_scaling_{clients_to_use}clients.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSaved combined plot: {output_file}")
