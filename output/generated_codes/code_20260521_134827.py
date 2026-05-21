import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Read the differential expression data directly
df = pd.read_csv('example_data/differential_expression.csv')

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Classify genes
conditions = [
    (df['log2FoldChange'] > 1) & (df['padj'] < 0.05),
    (df['log2FoldChange'] < -1) & (df['padj'] < 0.05)
]
choices = ['upregulated', 'downregulated']
df['regulation'] = np.select(conditions, choices, default='not_significant')

counts = df['regulation'].value_counts()
print('Counts per category:')
print(counts)

colors = {'upregulated': '#e74c3c', 'downregulated': '#3498db', 'not_significant': '#95a5a6'}
bar_colors = [colors.get(cat, '#95a5a6') for cat in counts.index]

plt.figure(figsize=(8, 6))
bars = plt.bar(counts.index, counts.values, color=bar_colors, edgecolor='black', linewidth=1.2)

for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(val),
             ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.title('Differentially Expressed Genes: Up vs Down', fontsize=16, fontweight='bold')
plt.xlabel('Regulation Status', fontsize=14)
plt.ylabel('Number of Genes', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_file = f'output/de_gene_counts_barplot_{timestamp}.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f'Saved plot to {out_file}')
plt.close()