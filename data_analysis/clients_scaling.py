"""
Analyze the effect of number of clients on model performance.
Generates plots showing accuracy and loss vs number of clients.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re
import argparse

# Set up paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# CLI
parser = argparse.ArgumentParser(description="Client scaling analysis")
parser.add_argument("--round", type=int, default=None, help="Round number to extract (default: use rounds from filename or max in CSV)")
parser.add_argument("--pattern", type=str, default="*rounds*.csv", help="Glob pattern to find result CSVs (default: '*rounds*.csv')")
args = parser.parse_args()

# Find CSV files matching pattern
csv_files = list(RESULTS_DIR.glob(args.pattern))

if not csv_files:
    print(f"Error: No CSV files found with pattern {args.pattern} in {RESULTS_DIR!s}")
    exit(1)

print(f"Found {len(csv_files)} files:")
for f in sorted(csv_files):
    print(f"  - {f.name}")

# Extract data from each file
results = []

for csv_file in csv_files:
    # Extract number of clients from filename (e.g., "10clients_5rounds_...")
    match = re.search(r'(\d+)clients', csv_file.name)
    if not match:
        print(f"Warning: Could not extract client count from {csv_file.name}")
        continue
    
    num_clients = int(match.group(1))
    
    # Load CSV
    df = pd.read_csv(csv_file)

    # Determine which round to use:
    # 1) If user passed --round, use that
    # 2) Else try to parse '<N>rounds' from filename
    # 3) Else use the max round present in the CSV
    target_round = None
    if args.round is not None:
        target_round = args.round
    else:
        m_round = re.search(r'_(\d+)rounds_', csv_file.name)
        if m_round:
            target_round = int(m_round.group(1))
        else:
            target_round = int(df['round'].max())

    # Safeguard: if the chosen target_round is not present, fall back to max round
    if target_round not in df['round'].unique():
        target_round = int(df['round'].max())

    # Get final round metrics
    train_rows = df[(df['round'] == target_round) & (df['phase'] == 'train')]
    eval_rows = df[(df['round'] == target_round) & (df['phase'] == 'evaluate')]

    if train_rows.empty or eval_rows.empty:
        # fallback to last available train/eval rows
        final_train = df[df['phase'] == 'train'].iloc[-1]
        final_eval = df[df['phase'] == 'evaluate'].iloc[-1]
        used_round = int(final_train['round'])
    else:
        final_train = train_rows.iloc[0]
        final_eval = eval_rows.iloc[0]
        used_round = target_round

    print(f"  Using round {used_round} for {csv_file.name}")
    
    results.append({
        'num_clients': num_clients,
        'train_accuracy': final_train['accuracy'],
        'train_loss': final_train['loss'],
        'eval_accuracy': final_eval['accuracy'],
        'eval_loss': final_eval['loss']
    })

# Convert to DataFrame and sort by number of clients
results_df = pd.DataFrame(results).sort_values('num_clients')

print("\n" + "="*60)
print("COLLECTED METRICS")
print("="*60)
print(results_df.to_string(index=False))
print("="*60)

# Create two-panel plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Client Scaling Analysis (25 Rounds)', fontsize=16, fontweight='bold')

x = results_df['num_clients']

# Plot 1: Accuracy
ax1.plot(x, results_df['train_accuracy'], 'o-', label='Train Accuracy', 
         color='#e74c3c', linewidth=2, markersize=8)
ax1.plot(x, results_df['eval_accuracy'], 's-', label='Eval Accuracy', 
         color='#3498db', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clients', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy vs Number of Clients', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max(results_df[['train_accuracy', 'eval_accuracy']].max()) * 1.1)

# Plot 2: Loss
ax2.plot(x, results_df['train_loss'], 'o-', label='Train Loss', 
         color='#e67e22', linewidth=2, markersize=8)
ax2.plot(x, results_df['eval_loss'], 's-', label='Eval Loss', 
         color='#9b59b6', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clients', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Loss vs Number of Clients', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / "client_scaling_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Client range: {results_df['num_clients'].min()} to {results_df['num_clients'].max()}")
print(f"\nBest Eval Accuracy: {results_df['eval_accuracy'].max():.4f} with {results_df.loc[results_df['eval_accuracy'].idxmax(), 'num_clients']:.0f} clients")
print(f"Worst Eval Accuracy: {results_df['eval_accuracy'].min():.4f} with {results_df.loc[results_df['eval_accuracy'].idxmin(), 'num_clients']:.0f} clients")
print(f"\nBest Eval Loss: {results_df['eval_loss'].min():.4f} with {results_df.loc[results_df['eval_loss'].idxmin(), 'num_clients']:.0f} clients")
print(f"Worst Eval Loss: {results_df['eval_loss'].max():.4f} with {results_df.loc[results_df['eval_loss'].idxmax(), 'num_clients']:.0f} clients")
print("="*60)
