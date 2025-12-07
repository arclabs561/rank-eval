#!/usr/bin/env python3
"""
Generate NDCG visualization charts using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

def dcg_at_k(relevance, k):
    """Compute DCG@k."""
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance[:k]))

def idcg_at_k(relevance, k):
    """Compute IDCG@k (ideal ranking)."""
    sorted_rel = sorted(relevance, reverse=True)
    return dcg_at_k(sorted_rel, k)

def ndcg_at_k(relevance, k):
    """Compute NDCG@k."""
    dcg = dcg_at_k(relevance, k)
    idcg = idcg_at_k(relevance, k)
    return dcg / idcg if idcg > 0 else 0.0

# 1. Discounting Function
fig, ax = plt.subplots(figsize=(10, 6))

ranks = np.arange(0, 20)
discounts = 1 / np.log2(ranks + 2)

ax.plot(ranks, discounts, marker='o', linewidth=2.5, markersize=6, 
        color='#00d9ff', markerfacecolor='#00d9ff', markeredgecolor='black', 
        markeredgewidth=1)
ax.fill_between(ranks, discounts, alpha=0.3, color='#00d9ff')

ax.set_xlabel('Rank Position (0-indexed)', fontweight='bold', fontsize=12)
ax.set_ylabel('Discount Factor: 1 / log₂(rank + 2)', fontweight='bold', fontsize=12)
ax.set_title('NDCG Discounting Function\nLower ranks contribute less to the final score', 
             fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 19.5)

# Add annotation for key points
ax.annotate('Rank 0: 1.0', xy=(0, 1.0), xytext=(3, 0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
ax.annotate('Rank 9: ~0.3', xy=(9, discounts[9]), xytext=(12, discounts[9] + 0.1),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / 'ndcg_discounting.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: ndcg_discounting.png")

# 2. Ranking Comparison: Good vs Poor
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Good ranking: relevant docs at top
ranking_good = [2, 0, 1, 0, 2, 0, 1, 0, 0, 0]  # rel=2 highly relevant, rel=1 relevant
ranking_poor = [0, 0, 0, 1, 0, 2, 0, 1, 0, 2]  # relevant docs at bottom

k_values = [1, 3, 5, 10]
ndcg_good = [ndcg_at_k(ranking_good, k) for k in k_values]
ndcg_poor = [ndcg_at_k(ranking_poor, k) for k in k_values]

x = np.arange(len(k_values))
width = 0.35

bars1 = ax1.bar(x - width/2, ndcg_good, width, label='Good Ranking', 
                color='#00ff88', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, ndcg_poor, width, label='Poor Ranking', 
                color='#ff4757', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('k (cutoff)', fontweight='bold', fontsize=12)
ax1.set_ylabel('NDCG@k', fontweight='bold', fontsize=12)
ax1.set_title('NDCG Comparison: Good vs Poor Ranking\nGood: relevant docs at top\nPoor: relevant docs at bottom', 
              fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels([f'NDCG@{k}' for k in k_values])
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1.0)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

# Right: Show the actual rankings
positions = np.arange(len(ranking_good))
ax2.barh(positions, ranking_good, color='#00ff88', alpha=0.8, 
         edgecolor='black', linewidth=1.5, label='Good Ranking')
ax2.barh(positions + 0.4, ranking_poor, color='#ff4757', alpha=0.8, 
         edgecolor='black', linewidth=1.5, label='Poor Ranking')

ax2.set_xlabel('Relevance Score', fontweight='bold', fontsize=12)
ax2.set_ylabel('Rank Position', fontweight='bold', fontsize=12)
ax2.set_title('Ranking Comparison\n(0=irrelevant, 1=relevant, 2=highly relevant)', 
              fontweight='bold', pad=15)
ax2.set_yticks(positions + 0.2)
ax2.set_yticklabels([f'Rank {i}' for i in positions])
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'ndcg_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: ndcg_comparison.png")

# 3. DCG Accumulation
fig, ax = plt.subplots(figsize=(10, 6))

# Good ranking
dcg_cumulative = []
for k in range(1, len(ranking_good) + 1):
    dcg_cumulative.append(dcg_at_k(ranking_good, k))

idcg_cumulative = []
for k in range(1, len(ranking_good) + 1):
    idcg_cumulative.append(idcg_at_k(ranking_good, k))

ndcg_cumulative = [dcg / idcg if idcg > 0 else 0 
                   for dcg, idcg in zip(dcg_cumulative, idcg_cumulative)]

k_range = np.arange(1, len(ranking_good) + 1)
ax.plot(k_range, dcg_cumulative, marker='o', linewidth=2.5, markersize=6,
        label='DCG@k', color='#00d9ff')
ax.plot(k_range, idcg_cumulative, marker='s', linewidth=2.5, markersize=6,
        label='IDCG@k (ideal)', color='#00ff88', linestyle='--')
ax.plot(k_range, ndcg_cumulative, marker='^', linewidth=2.5, markersize=6,
        label='NDCG@k', color='#ff6b9d')

ax.set_xlabel('k (cutoff)', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('DCG Accumulation: How NDCG Builds Up\nDCG accumulates relevance, IDCG is the ideal, NDCG normalizes', 
             fontweight='bold', pad=15)
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'ndcg_accumulation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: ndcg_accumulation.png")

print("\n✅ All NDCG visualizations generated!")

