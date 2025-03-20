from src.membrane_solver import MembraneSolver
import matplotlib.pyplot as plt
import numpy as np
from src.direct_method import SolveDirectMethod
import os
import seaborn as sns
from src.plot_eigenfrequencies import plot_eigenfrequencies_vs_L

sns.set(style="whitegrid")
plt.rc('text')
plt.rc('font', family='serif')

def solve_membrane():
    for shape in [ 'rectangle']:
        solver = MembraneSolver(n=30, shape=shape, use_sparse=True)
        solver.solve(num_modes=6)
        print(f"\nShape: {shape}, First 6 frequencies: {solver.frequencies}")
        solver.plot_modes()
        # for mode_idx in range(6):
        #     anim = solver.animate_mode(mode_idx=mode_idx, duration=5, fps=30, filename=f'animations/{shape}_mode_{mode_idx+1}.mp4')
        
        # solver.compare_performance()

def solve_direct_method():
    diffusion = SolveDirectMethod()
    diffusion.solve()
    #diffusion.animate()
    diffusion.plot()

if __name__ == "__main__":
    solve_membrane()
    # solve_direct_method()
    # plot_eigenfrequencies_vs_L()

