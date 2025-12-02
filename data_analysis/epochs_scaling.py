"""
Plot accuracy and loss vs number of local epochs using CSVs in results/epoch_tuning/

Saves: data_analysis/plots/epochs_scaling.png

CSV expected format: per-round rows with columns ['round','phase','loss','accuracy',...]
Filenames should include the epoch count, e.g. '5_epochs.csv' or '10_epochs.csv'
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse

# Paths
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'epoch_tuning'
OUTPUT_DIR = Path(__file__).parent / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description='Epochs scaling analysis')
parser.add_argument('--pattern', type=str, default='*_epochs.csv', help='Glob pattern to find epoch CSVs')
args = parser.parse_args()

csv_files = list(RESULTS_DIR.glob(args.pattern))
if not csv_files:
    print(f'No CSV files found in {RESULTS_DIR!s} with pattern {args.pattern}')
    raise SystemExit(1)

collected = []
for f in csv_files:
    m = re.search(r'(\d+)_?epochs', f.name)
    if not m:
        continue
    epochs = int(m.group(1))
    df = pd.read_csv(f)
    # take final round rows for train and evaluate
    try:
        tr = df[df['phase'] == 'train'].iloc[-1]
        er = df[df['phase'] == 'evaluate'].iloc[-1]
    except Exception:
        print(f'Skipping {f.name}: missing expected phase rows')
        continue

    collected.append({
        'epochs': epochs,
        'train_accuracy': float(tr['accuracy']),
        'eval_accuracy': float(er['accuracy']),
        'train_loss': float(tr['loss']),
        'eval_loss': float(er['loss']),
        'path': f.name
    })

if not collected:
    print('No valid epoch files found')
    raise SystemExit(1)

dfc = pd.DataFrame(collected).sort_values('epochs')
print('\nCollected:')
print(dfc[['epochs','train_accuracy','eval_accuracy','train_loss','eval_loss']].to_string(index=False))

# Plot accuracy and loss on two subplots in one PNG
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('Epochs Scaling Analysis', fontsize=14, fontweight='bold')

# Accuracy
ax1.plot(dfc['epochs'], dfc['train_accuracy'], 'o-', label='Train Accuracy', color='#e74c3c')
ax1.plot(dfc['epochs'], dfc['eval_accuracy'], 's-', label='Eval Accuracy', color='#3498db')
ax1.set_xlabel('Local Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs Local Epochs')
ax1.grid(alpha=0.3)
ax1.legend()
ax1.set_ylim(0, max(dfc[['train_accuracy','eval_accuracy']].max()) * 1.1)

# Loss
ax2.plot(dfc['epochs'], dfc['train_loss'], 'o-', label='Train Loss', color='#e67e22')
ax2.plot(dfc['epochs'], dfc['eval_loss'], 's-', label='Eval Loss', color='#9b59b6')
ax2.set_xlabel('Local Epochs')
ax2.set_ylabel('Loss')
ax2.set_title('Loss vs Local Epochs')
ax2.grid(alpha=0.3)
ax2.legend()
try:
    ax2.set_ylim(0, max(dfc[['train_loss','eval_loss']].max()) * 1.1)
except Exception:
    pass

plt.tight_layout()
outfile = OUTPUT_DIR / 'epochs_scaling.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()

print(f'\nSaved plot: {outfile}')
