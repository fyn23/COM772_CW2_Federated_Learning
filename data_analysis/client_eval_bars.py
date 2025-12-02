"""
Plot per-round evaluate accuracy and loss for all clients (grouped bars) and a focused single-client plot.

Reads JSON files in `results/client/` where each file is a JSON list of records with keys: phase, round, loss, accuracy.

Usage:
  python data_analysis/client_eval_bars.py [--client USER_ID]

If --client is omitted the script will pick the first client found.
"""

import json
from pathlib import Path
import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_CLIENT_DIR = Path(__file__).parent.parent / 'results' / 'client'
OUTPUT_DIR = Path(__file__).parent / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description='Client evaluate per-round plots')
parser.add_argument('--client', type=str, default=None, help='Client user_id to focus on')
args = parser.parse_args()

json_files = sorted(list(RESULTS_CLIENT_DIR.glob('*.json')))
if not json_files:
    print(f'No client JSON files found in {RESULTS_CLIENT_DIR!s}')
    raise SystemExit(1)

# Load all client JSONs into a DataFrame: columns = client, phase, round, loss, accuracy
rows = []
for jf in json_files:
    client_id = jf.stem
    try:
        with open(jf, 'r') as fh:
            data = json.load(fh)
    except Exception:
        continue
    for rec in data:
        if rec.get('phase') != 'evaluate':
            continue
        try:
            r = int(rec.get('round', -1))
            rows.append({'client': client_id, 'round': r, 'loss': float(rec.get('loss', np.nan)), 'accuracy': float(rec.get('accuracy', np.nan))})
        except Exception:
            continue

if not rows:
    print('No evaluate records found across clients')
    raise SystemExit(1)

df = pd.DataFrame(rows)

# If rounds are missing (e.g., all -1), assign sequential per-client round indices
if df['round'].le(0).all():
    # Create a sequential round index per client based on occurrence order
    seq_rows = []
    for client_id, group in df.groupby('client'):
        group = group.sort_index().reset_index(drop=True)
        for i, (_, row) in enumerate(group.iterrows(), start=1):
            seq_rows.append({'client': client_id, 'round': i, 'loss': row['loss'], 'accuracy': row['accuracy']})
    df = pd.DataFrame(seq_rows)

# Pivot to have rounds as index and clients as columns for accuracy and loss
acc_pivot = df.pivot_table(index='round', columns='client', values='accuracy')
loss_pivot = df.pivot_table(index='round', columns='client', values='loss')

# Choose client to focus on (random if not provided)
if args.client is None:
    focus_client = random.choice(list(acc_pivot.columns))
    print(f'Auto-selected random client for focus plot: {focus_client}')
else:
    focus_client = args.client
    if focus_client not in acc_pivot.columns:
        print(f'Client {focus_client} not found, available clients: {list(acc_pivot.columns)}')
        raise SystemExit(1)

# Plot grouped bars for accuracy across clients per round
rounds = acc_pivot.index.tolist()
n_rounds = len(rounds)
clients = list(acc_pivot.columns)
n_clients = len(clients)

fig = plt.figure(figsize=(16, 8))

# Subplot 1: grouped bars (accuracy) — for readability, limit to first 10 rounds if many
ax1 = fig.add_subplot(2, 1, 1)
max_rounds_display = min(n_rounds, 20)
display_rounds = rounds[:max_rounds_display]
indices = np.arange(len(display_rounds))
width = 0.8 / max(1, n_clients)

for i, client in enumerate(clients):
    vals = acc_pivot[client].reindex(display_rounds).fillna(0).values
    ax1.bar(indices + i*width, vals, width=width, label=client)

ax1.set_xticks(indices + width*(n_clients-1)/2)
ax1.set_xticklabels(display_rounds)
ax1.set_xlabel('Round')
ax1.set_ylabel('Evaluate Accuracy')
ax1.set_title('Evaluate Accuracy per Round — All Clients (first rounds shown)')
ax1.legend(ncol=min(4, n_clients), fontsize='small')
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: focused client evaluate accuracy across all rounds (line plot)
ax2 = fig.add_subplot(2, 1, 2)
focus_acc = acc_pivot[focus_client].reindex(rounds)
ax2.plot(rounds, focus_acc, 'o-', label=f'{focus_client} Eval Accuracy', color='#3498db', linewidth=2)
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Round')
ax2.set_title(f'Client {focus_client} — Evaluate Accuracy per Round')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, min(1.0, max(focus_acc.max() * 1.1, 1.0)))
ax2.legend()

plt.tight_layout()
outfile = OUTPUT_DIR / f'client_eval_all_and_{focus_client}.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()

print(f'Saved client evaluation plot: {outfile}')
