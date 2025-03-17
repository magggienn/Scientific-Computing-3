'''
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description:


The code also provides functions to animate the strings motion and plot time snapshots of the
string at various time steps.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
import time
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh

sns.set(style="whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

LABELSIZE = 14
TICKSIZE = 12


class MembraneSolver:
    def __init__(self, n, shape='square', L=1.0, use_sparse=False):
        self.n = n  # Grid size
        self.L = L  # Physical length
        self.shape = shape
        self.use_sparse = use_sparse
        self.dx = L / (n - 1)
        
        self.x = np.linspace(0, L, n)
        self.y = np.linspace(0, L, n)
        self.mask = np.ones((n, n), dtype=bool)
        
        if shape == 'circle':
            X, Y = np.meshgrid(self.x, self.y)
            R = L / 2  # Circle radius
            self.mask = (X - L/2) ** 2 + (Y - L/2) ** 2 <= R ** 2
        elif shape == 'rectangle':
            self.y = np.linspace(0, 2 * L, n)
        
        self.num_points = np.sum(self.mask)
        self.indices = np.full((n, n), -1, dtype=int)
        self.indices[self.mask] = np.arange(self.num_points)
        self.build_matrix()
    
    def build_matrix(self):
        n = self.n
        dx2 = self.dx ** 2
        data, rows, cols = [], [], []
        
        for i in range(n):
            for j in range(n):
                if self.mask[i, j]:
                    idx = self.indices[i, j]
                    data.append(-4 / dx2)
                    rows.append(idx)
                    cols.append(idx)
                    
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and self.mask[ni, nj]:
                            data.append(1 / dx2)
                            rows.append(idx)
                            cols.append(self.indices[ni, nj])
        
        self.A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(self.num_points, self.num_points))
    
    def solve(self, num_modes=6):
        if self.use_sparse:
            eigenvalues, eigenvectors = eigs(self.A, k=num_modes, which='SM', tol=1e-8)
        else:
            eigenvalues, eigenvectors = eigh(self.A.toarray())
            eigenvalues, eigenvectors = eigenvalues[:num_modes], eigenvectors[:, :num_modes]
        
        idx = np.argsort(np.abs(eigenvalues))
        self.eigenvalues, self.eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        self.frequencies = np.sqrt(np.maximum(0, -self.eigenvalues))
    
    def plot_modes(self, num_modes=6):
        n_cols = 2
        n_rows = int(np.ceil(num_modes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for i in range(num_modes):
            mode = np.zeros((self.n, self.n))
            mode[self.mask] = self.eigenvectors[:, i]
            ax = axes[i]
            im = ax.imshow(mode, cmap='Spectral', origin='lower', extent=[0, self.L, 0, self.L])
            ax.set_title(f'Mode {i+1}\nFreq: {self.frequencies[i]:.3f}')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if i % n_cols == 0:
                ax.set_ylabel('y', fontsize=LABELSIZE)
            else:
                cbar.set_label('Amplitude', fontsize=LABELSIZE)

            if i == num_modes - 1 or i == num_modes - 2:
                ax.set_xlabel('x', fontsize=LABELSIZE)
                ax.tick_params(labelbottom=True, labelsize=TICKSIZE)
        
        # Hide unused subplots
        for j in range(num_modes, n_rows * n_cols):
            if j < len(axes):
                axes[j].set_visible(False)
                
        plt.tight_layout()
        plt.savefig(f'figures/{self.shape}_modes.pdf')    
        plt.show()
    
    def compare_performance(self):
        start_dense = time.time()
        eigh(self.A.toarray())
        end_dense = time.time()
        
        start_sparse = time.time()
        eigs(self.A, k=6, which='SM', tol=1e-8)
        end_sparse = time.time()
        
        print(f"Dense solver time: {end_dense - start_dense:.4f}s")
        print(f"Sparse solver time: {end_sparse - start_sparse:.4f}s")
    
if __name__ == "__main__":
    # Test the solver for different shapes
    for shape in ['square', 'rectangle', 'circle']:
        solver = MembraneSolver(n=30, shape=shape, use_sparse=True)
        solver.solve(num_modes=6)
        print(f"\nShape: {shape}, First 6 frequencies: {solver.frequencies}")
        solver.plot_modes()
        solver.compare_performance()