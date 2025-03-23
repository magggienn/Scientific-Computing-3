"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Contains functions for plotting the laplacian matrices for the
different membrane shapes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from src.membrane_solver import MembraneSolver

sns.set(style="whitegrid")
plt.rc('text')
plt.rc('font', family='serif')
LABELSIZE = 40
TICKSIZE = 12


def visualize_matrix_structure(solver):
    """
    Visualizes the matrix structure for a single membrane shape
    """
    matrix = solver.A.toarray()
    size = min(50, matrix.shape[0])
    matrix_subset = matrix[:size, :size]

    plt.figure(figsize=(6, 5))
    plt.title(f"{solver.shape.capitalize()} Membrane Matrix", fontsize=LABELSIZE)

    # Create heatmap and capture the returned mappable object that contains the colorbar info
    heatmap = sns.heatmap(matrix_subset, cmap='Spectral', center=0)

    # Get and customize the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICKSIZE+10)

    plt.xlabel("Column Index", fontsize=LABELSIZE)
    plt.ylabel("Row Index", fontsize=LABELSIZE)
    plt.xticks(fontsize=TICKSIZE)
    plt.yticks(fontsize=TICKSIZE)

    filename = f"figures/boundaries/{solver.shape}_membrane_matrix_values.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def visualize_all_matrices_in_one_row():
    """
    Creates a single figure with all three membrane shapes side by side in one row
    """
    shapes = ['square', 'rectangle', 'circle']
    solvers = {shape: MembraneSolver(n=30, shape=shape, use_sparse=True) for shape in shapes}

    for shape, solver in solvers.items():
        solver.solve(num_modes=6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, shape in enumerate(shapes):
        matrix = solvers[shape].A.toarray()
        size = min(50, matrix.shape[0])
        matrix_subset = matrix[:size, :size]

        ax = axes[i]
        heatmap = sns.heatmap(matrix_subset, cmap='Spectral', center=0, ax=ax)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=TICKSIZE+10)

        ax.set_xlabel("Column Index", fontsize=LABELSIZE-2)
        ax.set_ylabel("Row Index", fontsize=LABELSIZE-2)
        ax.tick_params(labelsize=TICKSIZE)

    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = "figures/boundaries/all_membrane_matrices_comparison.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
