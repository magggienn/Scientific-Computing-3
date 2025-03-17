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
from matplotlib.animation import FuncAnimation
import seaborn as sns
import matplotlib.animation as animation
from scipy.sparse import linalg as sparse_linalg
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
import scipy.sparse

# Plotting parameters
sns.set(style="whitegrid") 
plt.rc('text', usetex=True) 
plt.rc('font', family='serif')
labelsize = 14
ticksize = 14
colors = sns.color_palette("Set2", 8)


class MembraneSolver:
    def __init__(self, n,c=1, shape='square', L=1.0, use_sparse=False):
        """
        Initialize the membrane solver.
        """
        self.n = n
        self.L = L
        self.shape = shape
        self.use_sparse = use_sparse
        self.c = c
        
        if shape == 'square':
            # Create a grid for the square membrane
            self. x = np.linspace(0, L, n)
            self.y = np.linspace(0, L, n)
            self.dx = self.x[1] - self.x[0] # the grid spacing between x points
            self.dy = self.y[1] - self.y[0] # the grid spacing between y points
            self.mask = np.ones((n, n), dtype=bool)
            
        # Create mapping from 2D grid to 1D indices for points inside the domain
        self.num_points = np.sum(self.mask) # number of points inside the domain
        self.indices = np.full((n, n), -1, dtype=int) # prepare indices that have -1 for points outside the domain
        self.indices[self.mask] = np.arange(self.num_points) # assign indices to points inside the domain
        
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequencies = None
        
        self.build_matrix()
        
    def build_matrix(self):
        """Build the Laplacian matrix for the selected shape with Dirichlet boundary conditions."""
        n = self.n
        dx = self.dx
        dy = self.dy
        
        # Initialize data for sparse matrix construction
        laplace_data = []
        row_indices = []
        col_indices = []
        
        # Fill the matrix with discretized Laplacian
        for i in range(n):
            for j in range(n):
                if self.mask[i, j]:
                    idx = self.indices[i, j]
                    
                    # Set diagonal element (center of stencil)
                    laplace_data.append(-4/(dx*dx))
                    row_indices.append(idx)
                    col_indices.append(idx)
                    
                    # Check neighbors
                    # (-1,0) is left, (1,0) is right, (0,-1) is down, (0,1) is up
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj # current neighbor
                        
                        if 0 <= ni < n and 0 <= nj < n:
                            if self.mask[ni, nj]:
                                # Interior neighbor
                                neighbor_idx = self.indices[ni, nj]
                                laplace_data.append(1/(dx*dx))
                                row_indices.append(idx)
                                col_indices.append(neighbor_idx)
                                                            
        # Create the sparse matrix
        if self.use_sparse:
            self.A = scipy.sparse.csr_matrix(
                (laplace_data, (row_indices, col_indices)), 
                shape=(self.num_points, self.num_points)
            )
        else:
            # Convert to dense matrix
            A_sparse = scipy.sparse.csr_matrix(
                (laplace_data, (row_indices, col_indices)), 
                shape=(self.num_points, self.num_points)
            )
            self.A = A_sparse.toarray()
        
        return self.A
        
    def solve(self, num_modes=6):
        """
        Solve the eigenvalue problem to find eigenmodes and eigenfrequencies.
        
        Returns:
        eigenvalues: The eigenvalues (related to frequencies)
        eigenvectors: The eigenvectors (eigenmodes)
        """
        if self.use_sparse:
            from scipy.sparse.linalg import eigsh
            # Use eigsh for symmetric matrices
            eigenvalues, eigenvectors = eigsh(self.A, k=num_modes, which='SM')
        else:
            from scipy.linalg import eigh
            # Use eigh for symmetric dense matrices
            eigenvalues, eigenvectors = eigh(self.A)
            # Keep only the first num_modes
            eigenvalues = eigenvalues[:num_modes]
            eigenvectors = eigenvectors[:, :num_modes]
        
        # Eigenvalues for the laplace operator should be negative
        # Sort by frequency (smallest absolute eigenvalue first)
        idx = np.argsort(np.abs(eigenvalues))
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        # Calculate frequencies
        self.frequencies = np.sqrt(np.maximum(0, -self.eigenvalues))
        
        return self.eigenvalues, self.eigenvectors
    
    def get_mode_grid(self, mode_idx=0):
        """
        Get the eigenmode as a 2D grid with zeros outside the domain.
        
        Parameters:
        mode_idx: Index of the eigenmode to retrieve
        
        Returns:
        2D grid representation of the eigenmode
        """
        if self.eigenvectors is None:
            self.solve()
        
        mode_grid = np.zeros((self.n, self.n))
        mode_values = self.eigenvectors[:, mode_idx]
        
        mode_grid[self.mask] = mode_values
        
        return mode_grid
    
    def plot_mode(self, mode_idx=0, figsize=(10, 8)):
        """
        Plot an eigenmode of the membrane.
        
        Parameters:
        mode_idx: Index of the eigenmode to plot
        figsize: Figure size (width, height)
        
        Returns:
        matplotlib figure
        """
        mode_grid = self.get_mode_grid(mode_idx)
        
        # Get coordinates for proper plotting
        X, Y = np.meshgrid(self.x, self.y)
        
        # Create masked array for circular domain
        if self.shape == 'circle':
            mode_grid = np.ma.masked_array(mode_grid, ~self.mask)
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(X, Y, mode_grid, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax, label='Amplitude')
        ax.set_title(f'Eigenmode {mode_idx}, Frequency: {self.frequencies[mode_idx]:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        return fig
                
        
    def animate_mode(self, mode_idx, duration=5, fps=30, filename=None, figsize=(8, 8)):
        """
        Animate a specific eigenmode showing its vibration over time.
        
        Args:
            mode_idx: Index of the mode to animate
            duration: Duration of the animation in seconds
            fps: Frames per second
            filename: If provided, the animation will be saved to this file
            figsize: Figure size (width, height) in inches
        """
        if not hasattr(self, 'eigenvectors') or not hasattr(self, 'frequencies'):
            raise ValueError("You need to call solve() before animating")
        
        # Calculate total number of frames
        num_frames = duration * fps
        
        # Get the frequency for this mode
        frequency = self.frequencies[mode_idx]
        
        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare the mode data
        if self.shape == 'circle':
            # For circle, map the eigenvector back to 2D grid
            mode_2d = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    if self.inside_points[i, j] >= 0:
                        mode_2d[i, j] = self.eigenvectors[self.inside_points[i, j], mode_idx]
            mode = mode_2d
            extent = [0, self.L, 0, self.L]
        else:
            # For square and rectangle
            mode = self.eigenvectors[:, mode_idx].reshape(self.n, self.n)
            if self.shape == 'square':
                extent = [0, self.L, 0, self.L]
            else:  # rectangle
                extent = [0, self.L, 0, 2*self.L]
        
        # Normalize the mode for better visualization
        mode = mode / np.max(np.abs(mode))
        
        # Find min/max values for consistent color scale
        vmax = np.max(np.abs(mode))
        vmin = -vmax
        
        # Create initial plot
        im = ax.imshow(mode, cmap='coolwarm', origin='lower', 
                    extent=extent, vmin=vmin, vmax=vmax)
        
        # Add colorbar and labels
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Amplitude', fontsize=14)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        title_obj = ax.set_title(f'Eigenmode {mode_idx+1}, Frequency: {frequency:.4f}', fontsize=14)
        
        def update(frame):
            """Update function for the animation."""
            # Calculate time point
            t = frame / fps
            
            # Calculate amplitude factor for this time point
            # We use cos to make the membrane oscillate
            amplitude = np.cos(2 * np.pi * frequency * t)
            
            # Apply amplitude to the mode
            displayed_mode = amplitude * mode
            
            # Update the plot
            im.set_array(displayed_mode)
            title_obj.set_text(f'Eigenmode {mode_idx+1}, Frequency: {frequency:.4f}, Time: {t:.2f}s')
            return [im, title_obj]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames,
                            interval=1000/fps, blit=False)
        
        # Save animation if filename is provided
        if filename:
            anim.save(filename, writer='ffmpeg', fps=fps)
            print(f"Animation saved to {filename}")
        
        plt.close() if filename else plt.tight_layout()
        return anim