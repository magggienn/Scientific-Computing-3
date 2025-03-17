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
import matplotlib.animation as animation
import os

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
            # Update dx for rectangular grid
            self.dx = L / (n - 1)  # x spacing
            self.dy = (2 * L) / (n - 1)  # y spacing
        else:  # square
            self.dx = L / (n - 1)  # Uniform spacing
        
        self.num_points = np.sum(self.mask)
        self.indices = np.full((n, n), -1, dtype=int)
        self.indices[self.mask] = np.arange(self.num_points)
        self.build_matrix()
    
    def build_matrix(self):
        """ Build the matrix for the eigenvalue problem """
        n = self.n
        data, rows, cols = [], [], []
        
        for i in range(n):
            for j in range(n):
                if self.mask[i, j]:
                    idx = self.indices[i, j]
                    
                    # For rectangle, use dx^2 for x direction and dy^2 for y direction
                    if self.shape == 'rectangle':
                        dx2 = self.dx ** 2
                        dy2 = self.dy ** 2
                        
                        # central coefficient depends on both dx and dy
                        central_coef = -2/dx2 - 2/dy2
                        data.append(central_coef)
                        rows.append(idx)
                        cols.append(idx)
                        
                        for di, dj, weight in [(-1, 0, 1/dx2), (1, 0, 1/dx2), 
                                            (0, -1, 1/dy2), (0, 1, 1/dy2)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < n and self.mask[ni, nj]:
                                data.append(weight)
                                rows.append(idx)
                                cols.append(self.indices[ni, nj])
                    else:
                        dx2 = self.dx ** 2
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
        """ Solve the eigenvalue problem """
        if self.use_sparse:
            eigenvalues, eigenvectors = eigs(self.A, k=num_modes, which='SM', tol=1e-8)
        else:
            eigenvalues, eigenvectors = eigh(self.A.toarray())
            eigenvalues, eigenvectors = eigenvalues[:num_modes], eigenvectors[:, :num_modes]
        
        idx = np.argsort(np.abs(eigenvalues))
        self.eigenvalues, self.eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        self.frequencies = np.sqrt(np.maximum(0, -self.eigenvalues))
    
    def plot_modes(self, num_modes=6):
        """ Plot the first num_modes eigenmodes """
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
        """ Compare the performance of dense and sparse solvers """
        start_dense = time.time()
        eigh(self.A.toarray())
        end_dense = time.time()
        
        start_sparse = time.time()
        eigs(self.A, k=6, which='SM', tol=1e-8)
        end_sparse = time.time()
        
        print(f"Dense solver time: {end_dense - start_dense:.4f}s")
        print(f"Sparse solver time: {end_sparse - start_sparse:.4f}s")
    
    def animate_mode(self, mode_idx=0, duration=10, fps=30, filename=None):
        '''
        Animate a specific eigenmode of the membrane
        '''
        mode_vector = self.eigenvectors[:, mode_idx]
        frequency = self.frequencies[mode_idx]
        
        mode = np.zeros((self.n, self.n))
        mode[self.mask] = mode_vector
        
        # Calculate total number of frames
        num_frames = int(duration * fps)
        
        # Time points for the animation
        t_max = 2 * np.pi / frequency  # One full period
        times = np.linspace(0, duration, num_frames)
        
        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Find the maximum amplitude for consistent color scaling
        max_amp = np.max(np.abs(mode))
        
        # Initial plot
        X, Y = np.meshgrid(self.x, self.y) if self.shape != 'rectangle' else np.meshgrid(self.x, self.y)
        
        # For 3D visualization
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X', fontsize=LABELSIZE)
        ax.set_ylabel('Y', fontsize=LABELSIZE)
        ax.set_zlabel('Amplitude', fontsize=LABELSIZE)
        ax.set_title(f'Eigenmode {mode_idx+1} Animation (Freq: {frequency:.3f})', fontsize=LABELSIZE)
        
        # Create the initial surface plot
        surf = ax.plot_surface(X, Y, np.zeros_like(mode), cmap='Spectral', 
                               vmin=-max_amp, vmax=max_amp, edgecolor='none')
        
        # Add a color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Amplitude', fontsize=LABELSIZE)
        
        # Set axis limits
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, 2*self.L if self.shape == 'rectangle' else self.L)
        ax.set_zlim(-max_amp, max_amp)
        
        def update(frame):
            ax.clear()
            
            # Calculate the displacement at this time
            t = times[frame]
            displacement = mode * np.cos(frequency * t)
            
            # Update the surface
            surf = ax.plot_surface(X, Y, displacement, cmap='Spectral', 
                                  vmin=-max_amp, vmax=max_amp, edgecolor='none')
            
            # Reset labels and title
            ax.set_xlabel('X', fontsize=LABELSIZE)
            ax.set_ylabel('Y', fontsize=LABELSIZE)
            ax.set_zlabel('Amplitude', fontsize=LABELSIZE)
            ax.set_title(f'Eigenmode {mode_idx+1} Animation (Freq: {frequency:.3f})', fontsize=LABELSIZE)
            
            # Set axis limits
            ax.set_xlim(0, self.L)
            ax.set_ylim(0, 2*self.L if self.shape == 'rectangle' else self.L)
            ax.set_zlim(-max_amp, max_amp)
            
            return [surf]
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=False)
        
        if filename:
            # Make sure the animations directory exists
            os.makedirs('animations', exist_ok=True)
            
            # Determine the writer based on the file extension
            if filename.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            elif filename.endswith('.gif'):
                writer = animation.PillowWriter(fps=fps)
            else:
                filename = f'animations/{self.shape}_mode_{mode_idx+1}.mp4'
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            
            anim.save(filename, writer=writer)
            print(f"Animation saved to {filename}")
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def plot_L(self):
        self.frequencies 
    
    
if __name__ == "__main__":
    # Test the solver for different shapes
    for shape in ['square', 'rectangle', 'circle']:
        solver = MembraneSolver(n=30, shape=shape, use_sparse=True)
        solver.solve(num_modes=6)
        print(f"\nShape: {shape}, First 6 frequencies: {solver.frequencies}")
        solver.plot_modes()
        solver.compare_performance()