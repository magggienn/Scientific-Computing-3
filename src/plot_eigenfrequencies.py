from src.membrane_solver import MembraneSolver
import matplotlib.pyplot as plt
import numpy as np
from src.direct_method import SolveDirectMethod
import os
import seaborn as sns
from matplotlib.lines import Line2D

sns.set(style="whitegrid")
plt.rc('text')
plt.rc('font', family='serif')
LABELSIZE = 14
TICKSIZE = 12

def plot_eigenfrequencies_vs_L(num_modes=6):
    """Plot eigenfrequencies as a function of L for different shapes"""    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    L_values = np.linspace(0.5, 3.0, 10)
    results = {}
    shapes = ['square', 'rectangle', 'circle']
    
    # Calculate eigenfrequencies for each shape and L value
    for shape in shapes:
        shape_freqs = []
        
        for L in L_values:
            solver = MembraneSolver(n=30, shape=shape, L=L, use_sparse=True)
            solver.solve(num_modes=num_modes)
            shape_freqs.append(solver.frequencies)
        
        results[shape] = np.array(shape_freqs)
        # for mode in range(num_modes):
        #         print(f"Mode {mode+1}: {results[shape][0, mode]:.4f}")
    
    # Create the plot with a clear white background 
    plt.figure(figsize=(10, 6), facecolor='white')
    
    # Use colors from the Spectral colormap for consistency with the mode visualization
    cmap = plt.cm.Spectral
    colors = {
        'square': cmap(0.93),      # Red end of Spectral
        'rectangle': cmap(0.76),   # Yellow/green middle of Spectral
        'circle': cmap(0.1)       # Blue end of Spectral
    }
    
    # Use markers that match the shapes
    markers = {'square': 's', 'rectangle': 'd', 'circle': 'o'}
    
    # Create a list to store legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Plot each shape and mode
    for i, shape in enumerate(shapes):
        # Plot the first mode with a solid line and markers
        line = plt.plot(L_values, results[shape][:, 0],
             color=colors[shape], 
             marker=markers[shape],
             markersize=8,
             linewidth=2.5,
             label=f"{shape.capitalize()}")[0]
        
        # Add to legend
        legend_handles.append(line)
        legend_labels.append(f"{shape.capitalize()}")
        
        # Plot higher modes with dashed lines
        for mode in range(1, num_modes):
            # Apply a small offset for square modes to make them visible
            offset = 0.02 * results[shape][:, mode] if shape == 'square' else 0
            
            plt.plot(L_values, results[shape][:, mode] + offset, 
                    color=colors[shape], 
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7)
            
            # Add mode number for identification
            if mode < 3:  # Limit annotations to avoid clutter
                plt.text(L_values[-1] * 1.02, results[shape][-1, mode], 
                        f"{mode+1}", color=colors[shape], fontweight='bold')
    
    # Add reference curve for 1/L scaling
    reference_L = np.linspace(L_values[0], L_values[-1], 100)
    reference_val = results['square'][0, 0] * L_values[0]  # Use first square mode as reference
    ref_line = plt.plot(reference_L, reference_val / reference_L, 'k--', linewidth=2)[0]
    
    # Add to legend
    legend_handles.append(ref_line)
    legend_labels.append("$\\sim 1/L$")
    
    # Create mode information legend elements
    solid_line = Line2D([0], [0], color='k', lw=2.5)
    dashed_line = Line2D([0], [0], color='k', lw=1.5, linestyle='--', alpha=0.7)
    
    # Add to legend
    legend_handles.extend([solid_line, dashed_line])
    legend_labels.extend(['Mode 1 (Fundamental)', 'Higher Modes (2-6)'])
    
    # Add labels
    plt.xlabel('L (Size Parameter)', fontsize=LABELSIZE)
    plt.ylabel('Eigenfrequency', fontsize=LABELSIZE)
    plt.title('Membrane Eigenfrequencies vs. Size (L)', fontsize=LABELSIZE)
    plt.tick_params(labelsize=TICKSIZE)
    plt.grid(True, alpha=0.3)
    
    # Create a single combined legend
    plt.legend(legend_handles, legend_labels, loc='upper right', fontsize=TICKSIZE)
    
    # Save figure
    plt.savefig('figures/eigenfrequencies_vs_L.pdf', dpi=300, bbox_inches='tight')
    print("Figure saved to figures/eigenfrequencies_vs_L.pdf")
    
    # Show the plot
    plt.show()
    
    return results