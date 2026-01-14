#!/usr/bin/env python3
"""
Create a standalone legend figure for convergence plots.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_legend_figure(output_path):
    """Create a legend-only figure with colored lines."""
    
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    
    # Create legend entries with actual colors and line styles
    legend_elements = [
        plt.Line2D([0], [0], color='#4e79a7', linewidth=2, label='DeGroot Baseline'),
        plt.Line2D([0], [0], color='#f28e2b', linewidth=1, label='Nano, A vs B'),
        plt.Line2D([0], [0], color='#59a14f', linewidth=1, label='Nano, B vs A'),
        plt.Line2D([0], [0], color='#e15759', linewidth=1, label='Mini, A vs B'),
        plt.Line2D([0], [0], color='#76b7b2', linewidth=1, label='Mini, B vs A'),
    ]
    
    # Create legend
    legend = ax.legend(handles=legend_elements, loc='center', 
                      ncol=3, frameon=True, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved legend figure to {output_path}")

if __name__ == "__main__":
    import sys
    import os
    
    output_path = "paper/figures/convergence_legend.png"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_legend_figure(output_path)
