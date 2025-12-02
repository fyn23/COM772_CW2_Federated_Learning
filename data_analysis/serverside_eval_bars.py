"""
Plot per-round evaluate accuracy and loss as bar charts (ignore training phase).

Usage:
  python data_analysis/eval_bars.py [--csv path/to/metrics.csv]
If --csv is omitted, the script picks the most recent CSV in `results/`.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

RESULTS_DIR = Path(__file__).parent.parent / 'results'
OUTPUT_DIR = Path(__file__).parent / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)


# Always use the provided CSV file unless --csv is specified
DEFAULT_CSV = RESULTS_DIR / 'iidvsniid' / '20clients_80rounds_niid_20251124_140725_metrics.csv'
parser = argparse.ArgumentParser(description='Per-round evaluate bars')
parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV), help='Metrics CSV to read')
args = parser.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    print(f'CSV not found: {csv_path}')
    raise SystemExit(1)

df = pd.read_csv(csv_path)
run_name = csv_path.stem

# Filter to evaluate phase only
eval_df = df[df['phase'] == 'evaluate'].copy()
if eval_df.empty:
    print('No evaluate phase rows found in the CSV')
    raise SystemExit(1)

# Sort by round
eval_df = eval_df.sort_values('round')

rounds = eval_df['round'].astype(int).tolist()
acc = eval_df['accuracy'].astype(float).tolist()
loss = eval_df['loss'].astype(float).tolist()

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f'Evaluate Metrics per Round — {run_name}', fontsize=14, fontweight='bold')

# Accuracy bar chart
axes[0].bar(rounds, acc, color='#3498db')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Evaluate Accuracy per Round')
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim(0, max(acc) * 1.1)

# Loss bar chart
axes[1].bar(rounds, loss, color='#e67e22')
axes[1].set_xlabel('Round')
axes[1].set_ylabel('Loss')
axes[1].set_title('Evaluate Loss per Round')
axes[1].grid(axis='y', alpha=0.3)
try:
    axes[1].set_ylim(0, max(loss) * 1.1)
except Exception:
    pass

plt.tight_layout()
outfile = OUTPUT_DIR / f'eval_bars_{run_name}.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()

print(f'Saved evaluate bars plot: {outfile}')
