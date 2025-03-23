from src.membrane_solver import MembraneSolver
import matplotlib.pyplot as plt
import numpy as np
from src.direct_method import SolveDirectMethod
import os
import seaborn as sns
import scipy.sparse
from src.plot_functions import plot_eigenfrequencies_vs_L, plot_combined_performance
from src.plot_matrix_M import visualize_all_matrices_in_one_row, visualize_matrix_structure

sns.set(style="whitegrid")
plt.rc('text')
plt.rc('font', family='serif')

def solve_membrane():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    shapes = ['square', 'rectangle', 'circle']
    
    results = {}
    for shape in ['square', 'rectangle', 'circle']:
        solver = MembraneSolver(n=30, shape=shape, use_sparse=True)
        # Solve to compare performance sparse and not sparse
        dense_stats, sparse_stats = solver.compare_performance()
        results[shape] = {'dense': dense_stats, 'sparse': sparse_stats}
        solver.solve(num_modes=6)
        solver.plot_modes()
        # visualize_matrix_structure(solver)
        # for mode_idx in range(6):
        #     anim = solver.animate_mode(
        #         mode_idx=mode_idx, 
        #         duration=5, 
        #         fps=30, 
        #         filename=f'animations/{shape}/{shape}_mode_{mode_idx+1}.mp4'
            # )
    #plot_combined_performance(results, n_value=30)

def solve_direct_method():
    diffusion = SolveDirectMethod()
    diffusion.solve()
    diffusion.plot()

if __name__ == "__main__":
    # solve_membrane()
    # solve_direct_method()
    plot_eigenfrequencies_vs_L()