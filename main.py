"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Combines all the different modules into one and runs the required
parts and plots.

Typical usage example:

python3 main.py
"""
from src.membrane_solver import MembraneSolver
import matplotlib.pyplot as plt
import numpy as np
from src.direct_method import SolveDirectMethod
import os
import seaborn as sns
import scipy.sparse
from src.plot_functions import plot_eigenfrequencies_vs_L, plot_combined_performance
from src.plot_matrix_M import visualize_all_matrices_in_one_row, visualize_matrix_structure
from src.leapfrog import (LeapfrogHarmonicOscillator, calculate_natural_frequency,
                           compare_energy_conservation, plot_phase_space_comparison)

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
        # Uncomment to make animation
        # visualize_matrix_structure(solver)
        # for mode_idx in range(6):
        #     anim = solver.animate_mode(
        #         mode_idx=mode_idx,
        #         duration=5,
        #         fps=30,
        #         filename=f'animations/{shape}/{shape}_mode_{mode_idx+1}.mp4'
            # )
    plot_combined_performance(results, n_value=30)


def solve_direct_method():
    diffusion = SolveDirectMethod()
    diffusion.solve()
    diffusion.plot()


def plot_leapfrog():
    # plotting of position/velocity and energy conservation
    k = 1.0
    dt = 0.001
    t_max = 20
    lf_oscillator = LeapfrogHarmonicOscillator(k=k, dt=dt, t_max=t_max)
    lf_oscillator.plot_results(x0=1.0, v0=0.0, ks=[0.5, 1.0, 2.0])

    # enerby conservation
    compare_energy_conservation(x0=1.0, v0=0.0, k=1.0, dt=0.01, t_max=50)


    cf = calculate_natural_frequency(k=1.0)
    plot_phase_space_comparison(x0=1.0, v0=0.0,
                                frequencies_list=[[0.4 * cf, cf, 1.6 * cf],
                                                  [cf]], rf=cf)


if __name__ == "__main__":
    # Uncomment to run the membrane solver
    # solve_membrane()

    # Uncomment to run the direct method
    # solve_direct_method()

    # Uncomment to plot the eigenfrequencies as a function of L for different shapes
    # plot_eigenfrequencies_vs_L()

    # Uncomment to get all the leapfrog figures (also stored in figres/frog/..
    # plot_leapfrog()
    pass
