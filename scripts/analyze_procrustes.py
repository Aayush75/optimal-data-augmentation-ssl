"""
Analyze Procrustes distance during training to verify optimality.
Recreates Figure 4 from the paper.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("Loading experimental results...")

procrustes_to_target = np.load('results/procrustes_to_target.npy')
procrustes_to_random = np.load('results/procrustes_to_random.npy')

print(f"Loaded {len(procrustes_to_target)} training iterations")
print(f"Final distance to target: {procrustes_to_target[-1]:.4f}")
print(f"Final distance to random: {procrustes_to_random[-1]:.4f}")

improvement = (procrustes_to_random[0] - procrustes_to_target[-1]) / procrustes_to_random[0] * 100
print(f"Improvement over random: {improvement:.1f}%")

print("\nPlotting Procrustes distance over training...")

plt.figure(figsize=(14, 7))

plt.plot(procrustes_to_target, linewidth=2.5, label='Distance to Target', color='#2E86AB')
plt.plot(procrustes_to_random, linewidth=2.5, label='Distance to Random', color='#A23B72', linestyle='--')

plt.xlabel('Training Iteration', fontsize=14, fontweight='bold')
plt.ylabel('Average Procrustes Distance', fontsize=14, fontweight='bold')
plt.title('Procrustes Distance During Barlow Twins Training\n(Recreating Figure 4 from Paper)', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('results/plots', exist_ok=True)
plt.savefig('results/plots/procrustes_analysis.png', dpi=150, bbox_inches='tight')
print("Saved to results/plots/procrustes_analysis.png")

plt.figure(figsize=(10, 6))
window = 50
smoothed_target = np.convolve(procrustes_to_target, np.ones(window)/window, mode='valid')
smoothed_random = np.convolve(procrustes_to_random, np.ones(window)/window, mode='valid')

plt.plot(smoothed_target, linewidth=2, label='To Target (smoothed)', color='#2E86AB')
plt.plot(smoothed_random, linewidth=2, label='To Random (smoothed)', color='#A23B72', linestyle='--')

plt.xlabel('Training Iteration', fontsize=12)
plt.ylabel('Smoothed Procrustes Distance', fontsize=12)
plt.title(f'Smoothed Procrustes Distance (window={window})', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('results/plots/procrustes_smoothed.png', dpi=150, bbox_inches='tight')
print("Saved smoothed version to results/plots/procrustes_smoothed.png")

convergence_threshold = procrustes_to_target[-1] * 1.01
converged_at = np.where(procrustes_to_target <= convergence_threshold)[0]
if len(converged_at) > 0:
    convergence_iteration = converged_at[0]
    print(f"\nConverged at iteration {convergence_iteration} (within 1% of final value)")
else:
    print("\nDid not converge within 1% threshold")

print("\nStatistics:")
print(f"  Initial distance to target: {procrustes_to_target[0]:.4f}")
print(f"  Final distance to target: {procrustes_to_target[-1]:.4f}")
print(f"  Total reduction: {procrustes_to_target[0] - procrustes_to_target[-1]:.4f}")
print(f"  Average distance to random: {np.mean(procrustes_to_random):.4f}")

plt.show()
