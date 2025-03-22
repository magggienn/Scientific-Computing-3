from src.membrane_solver import MembraneSolver
import matplotlib.pyplot as plt
import numpy as np
from src.direct_method import SolveDirectMethod
import os
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sns.set(style="whitegrid")
colors = sns.color_palette("Set2", 4)
plt.rc('text')
plt.rc('font', family='serif')
LABELSIZE = 29 #24
TICKSIZE = 25 #20

def plot_eigenfrequencies_vs_L(num_modes=6):
    """Plot eigenfrequencies as a function of L for different shapes."""    
    os.makedirs('figures', exist_ok=True)
    
    L_values = np.linspace(0.5, 3.0, 10)
    results = {}
    shapes = ['square', 'rectangle', 'circle']
    
    for shape in shapes:
        shape_freqs = []
        
        for L in L_values:
            solver = MembraneSolver(n=30, shape=shape, L=L, use_sparse=True)
            solver.solve(num_modes=num_modes)
            shape_freqs.append(solver.frequencies)
        
        results[shape] = np.array(shape_freqs)

    plt.figure(figsize=(10, 6), facecolor='white')
    
    shape_colors = {
        'square': colors[0],
        'rectangle': colors[1],
        'circle': colors[2]
    }
    
    markers = {'square': 's', 'rectangle': 'd', 'circle': 'o'}
    
    legend_handles = []
    legend_labels = []
    
    for i, shape in enumerate(shapes):
        # Plot the first mode with a solid line and markers
        line = plt.plot(L_values, results[shape][:, 0],
             color=shape_colors[shape], 
             marker=markers[shape],
             markersize=8,
             linewidth=2.5,
             label=f"{shape.capitalize()}")[0]
        
        legend_handles.append(line)
        legend_labels.append(f"{shape.capitalize()}")
        
        # Plot higher modes with .- lines
        for mode in range(1, num_modes):
            # Apply small offset for square modes to improve visibility
            offset = 0.02 * results[shape][:, mode] if shape == 'square' else 0
            
            plt.plot(L_values, results[shape][:, mode] + offset, 
                    color=shape_colors[shape], 
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.8)
            
            if mode < 3:  # Limit annotations to avoid clutter
                plt.text(L_values[-1] * 1.02, results[shape][-1, mode], 
                        f"{mode+1}", color=shape_colors[shape], fontweight='bold')
    
    # Add reference curve for 1/L scaling
    reference_L = np.linspace(L_values[0], L_values[-1], 100)
    reference_val = results['square'][0, 0] * L_values[0]
    ref_line = plt.plot(reference_L, reference_val / reference_L, 'k--', linewidth=2, c = colors[3])[0]
    
    legend_handles.append(ref_line)
    legend_labels.append("$\\sim 1/L$")
    
    solid_line = Line2D([0], [0], color='k', lw=2.5)
    dashed_line = Line2D([0], [0], color='k', lw=1.5, linestyle='--', alpha=0.7)
    
    legend_handles.extend([solid_line, dashed_line])
    legend_labels.extend(['Mode 1 (Fundamental)', 'Higher Modes (2-6)'])
    
    plt.xlabel('L (Size Parameter)', fontsize=LABELSIZE)
    plt.ylabel('Eigenfrequency', fontsize=LABELSIZE)
    plt.title('Membrane Eigenfrequencies vs. Size (L)', fontsize=LABELSIZE)
    plt.tick_params(labelsize=TICKSIZE)
    plt.grid(True, alpha=0.3)

    plt.legend(legend_handles, legend_labels, loc='upper right', fontsize=TICKSIZE)
    
    plt.savefig('figures/eigenfrequencies_vs_L.pdf', dpi=300, bbox_inches='tight')
    print("Figure saved to figures/eigenfrequencies_vs_L.pdf")
    
    plt.show()
    
    return results

def plot_combined_performance(results_dict, n_value, save=True):
    """Create bar chart comparing performance between dense and sparse matrices."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    bar_width = 0.35
    shapes = list(results_dict.keys())
    x_positions = np.arange(len(shapes))
    
    dense_means = [results_dict[shape]['dense']['mean'] for shape in shapes]
    dense_stds = [results_dict[shape]['dense']['std'] for shape in shapes]
    sparse_means = [results_dict[shape]['sparse']['mean'] for shape in shapes]
    sparse_stds = [results_dict[shape]['sparse']['std'] for shape in shapes]
    
    dense_bars = ax.bar(x_positions - bar_width/2, dense_means, bar_width, 
                        yerr=dense_stds, capsize=10, label='Dense', 
                        color=colors[0], alpha=0.7)
    sparse_bars = ax.bar(x_positions + bar_width/2, sparse_means, bar_width, 
                         yerr=sparse_stds, capsize=10, label='Sparse', 
                         color=colors[1], alpha=0.7)
    
    ax.set_xlabel('Membrane Shape', fontsize=LABELSIZE)
    ax.set_ylabel('Execution Time (s)', fontsize=LABELSIZE)
    #ax.set_title(f'Performance Comparison by Shape (n={n_value})', fontsize=LABELSIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([s.capitalize() for s in shapes], fontsize=TICKSIZE)
    ax.tick_params(axis='y', labelsize=TICKSIZE)
    ax.legend(fontsize=LABELSIZE-6, loc='center left')
    
    y_max = max(max(dense_means) + max(dense_stds), max(sparse_means) + max(sparse_stds)) * 1.2
    ax.set_ylim(0, y_max)
    
    for bar, mean, std in zip(dense_bars, dense_means, dense_stds):
        height = mean + std + 0.02*y_max
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.4f}s\n±{std:.4f}s', 
               ha='center', va='bottom', fontsize=TICKSIZE-4)
    
    for bar, mean, std in zip(sparse_bars, sparse_means, sparse_stds):
        height = mean + std + 0.02*y_max
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.4f}s\n±{std:.4f}s', 
               ha='center', va='bottom', fontsize=TICKSIZE-4)
    
    speedup_color = colors[3]  
    speedup_x_offset = 0.223
    for i, shape in enumerate(shapes):
        speedup = results_dict[shape]['dense']['mean'] / results_dict[shape]['sparse']['mean']
        
        # Create text with styled box to show speedup
        text = ax.text(i + speedup_x_offset, y_max * 0.77, f'Speedup: {speedup:.2f}x', 
                ha='center', va='center', fontsize=TICKSIZE-4,
                bbox=dict(
                    boxstyle="round,pad=0.4,rounding_size=0.5",
                    facecolor=speedup_color,
                    alpha=0.3,
                    edgecolor=speedup_color,
                    linewidth=2,
                    linestyle='-'
                ))
    
    if save:
        os.makedirs('figures/performance', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'figures/performance/comparison_all_shapes_n{n_value}.pdf')
        print(f"Performance comparison saved to figures/performance/comparison_all_shapes_n{n_value}.pdf")
    
    plt.tight_layout()
    plt.show()
    return fig