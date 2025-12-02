"""
Compare FedProx vs FedAvg using max evaluate accuracy and min evaluate loss.

Looks for CSVs in `results/FedProxVFedAvg/` and aggregates per-file metrics.
Saves `data_analysis/plots/fedprox_vs_fedavg.png`.
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'FedProxVFedAvg'
OUTPUT_DIR = Path(__file__).parent / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

csv_files = sorted(list(RESULTS_DIR.glob('*.csv')))
if not csv_files:
    print(f'No CSV files found in {RESULTS_DIR!s}')
    raise SystemExit(1)

records = []
for f in csv_files:
    df = pd.read_csv(f)
    eval_df = df[df['phase'] == 'evaluate']
    if eval_df.empty:
        continue
    max_acc = float(eval_df['accuracy'].max())
    min_loss = float(eval_df['loss'].min())
    name = f.stem
    # determine strategy and iid/niid from filename
    strategy = 'FedAvg' if 'FedAvg' in name else 'FedProx'
    data_type = 'IID' if '_iid' in name else 'non-IID'
    records.append({'file': f.name, 'strategy': strategy, 'data_type': data_type, 'max_eval_acc': max_acc, 'min_eval_loss': min_loss})

if not records:
    print('No evaluation records found')
    raise SystemExit(1)

df_rec = pd.DataFrame(records)
print(df_rec)

# Prepare grouped bar charts: for IID and non-IID, show two bars (FedAvg, FedProx) for acc and loss
groups = ['IID', 'non-IID']
strategies = ['FedAvg', 'FedProx']

acc_vals = []
loss_vals = []
for g in groups:
    row_acc = []
    row_loss = []
    for s in strategies:
        sel = df_rec[(df_rec['data_type'] == g) & (df_rec['strategy'] == s)]
        if sel.empty:
            row_acc.append(np.nan)
            row_loss.append(np.nan)
        else:
            row_acc.append(float(sel['max_eval_acc'].iloc[0]))
            row_loss.append(float(sel['min_eval_loss'].iloc[0]))
    acc_vals.append(row_acc)
    loss_vals.append(row_loss)

acc_vals = np.array(acc_vals)
loss_vals = np.array(loss_vals)

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Accuracy plot
x = np.arange(len(groups))
width = 0.35
axes[0].bar(x - width/2, acc_vals[:,0], width, label='FedAvg', color='#3498db')
axes[0].bar(x + width/2, acc_vals[:,1], width, label='FedProx', color='#e74c3c')
axes[0].set_xticks(x)
axes[0].set_xticklabels(groups)
axes[0].set_ylabel('Max Evaluate Accuracy')
axes[0].set_title('Max Evaluate Accuracy: FedAvg vs FedProx')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Loss plot
axes[1].bar(x - width/2, loss_vals[:,0], width, label='FedAvg', color='#3498db')
axes[1].bar(x + width/2, loss_vals[:,1], width, label='FedProx', color='#e74c3c')
axes[1].set_xticks(x)
axes[1].set_xticklabels(groups)
axes[1].set_ylabel('Min Evaluate Loss')
axes[1].set_title('Min Evaluate Loss: FedAvg vs FedProx')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
outfile = OUTPUT_DIR / 'fedprox_vs_fedavg.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()

print(f'Saved comparison plot: {outfile}')
