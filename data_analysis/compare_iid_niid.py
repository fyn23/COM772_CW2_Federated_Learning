"""
Compare IID vs non-IID federated learning results.
Generates bar charts and line plots for comparing training statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "iidvsniid"
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Find CSV files
all_csv_files = list(RESULTS_DIR.glob("*.csv"))
niid_files = [f for f in all_csv_files if '_niid' in f.name or f.name.endswith('niid.csv')]
iid_files = [f for f in all_csv_files if '_iid' in f.name and '_niid' not in f.name]

if not niid_files or not iid_files:
    print("Error: Could not find both IID and non-IID CSV files")
    print(f"NIID files found: {niid_files}")
    print(f"IID files found: {iid_files}")
    exit(1)

# Load most recent files (sorted by name, which includes timestamp)
niid_file = sorted(niid_files)[-1]
iid_file = sorted(iid_files)[-1]

print(f"Loading non-IID data from: {niid_file.name}")
print(f"Loading IID data from: {iid_file.name}")

# Load data
niid_df = pd.read_csv(niid_file)
iid_df = pd.read_csv(iid_file)

# Separate train and evaluate phases
niid_train = niid_df[niid_df['phase'] == 'train']
niid_eval = niid_df[niid_df['phase'] == 'evaluate']
iid_train = iid_df[iid_df['phase'] == 'train']
iid_eval = iid_df[iid_df['phase'] == 'evaluate']

# Calculate final statistics for bar chart
stats = {
    'Final Train Accuracy': [niid_train['accuracy'].iloc[-1], iid_train['accuracy'].iloc[-1]],
    'Final Eval Accuracy': [niid_eval['accuracy'].iloc[-1], iid_eval['accuracy'].iloc[-1]],
    'Final Train Loss': [niid_train['loss'].iloc[-1], iid_train['loss'].iloc[-1]],
    'Final Eval Loss': [niid_eval['loss'].iloc[-1], iid_eval['loss'].iloc[-1]],
}

# Create bar chart comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IID vs non-IID Federated Learning Comparison', fontsize=16, fontweight='bold')

x = np.arange(2)
width = 0.35
colors = ['#e74c3c', '#3498db']  # Red for non-IID, Blue for IID
labels = ['non-IID', 'IID']

# Plot each metric
metrics = [
    ('Final Train Accuracy', 0, 0, True),
    ('Final Eval Accuracy', 0, 1, True),
    ('Final Train Loss', 1, 0, False),
    ('Final Eval Loss', 1, 1, False)
]

for metric_name, row, col, is_accuracy in metrics:
    ax = axes[row, col]
    values = stats[metric_name]
    bars = ax.bar(x, values, width, color=colors)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(metric_name, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Set y-axis limits
    if is_accuracy:
        ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, max(values) * 1.2)

plt.tight_layout()
output_file = OUTPUT_DIR / "bar_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved bar chart to: {output_file}")
plt.close()

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for metric, values in stats.items():
    print(f"\n{metric}:")
    print(f"  non-IID: {values[0]:.4f}")
    print(f"  IID:     {values[1]:.4f}")
    diff = values[1] - values[0]
    pct = (diff / values[0]) * 100 if values[0] != 0 else 0
    print(f"  Difference: {diff:+.4f} ({pct:+.2f}%)")

print("\n" + "="*60)
print(f"Plot saved to: {OUTPUT_DIR / 'bar_comparison.png'}")
print("="*60)
