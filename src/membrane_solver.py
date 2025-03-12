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

# Plotting parameters
sns.set(style="whitegrid")  # Use seaborn style
plt.rc('text', usetex=True)  # Disable LaTeX to avoid missing dependency issues
plt.rc('font', family='serif')
labelsize = 14
ticksize = 14
colors = sns.color_palette("Set2", 8)


class MembraneSolver:
    def __init__(self, n, L, shape, use_sparse=False, c=1, boundary_condition=0):
        """
        shape: 'square', 'rectangle', or 'circle'
        L: Size parameter (side length for square, diameter for circle)
        n: Number of grid points in each dimension- interior points - not including boundary points 
        use_sparse (bool): Whether to use sparse matrix methods
        """
        self.n = n
        self.L = L
        self.shape = shape
        self.use_sparse = use_sparse
        self.c = c
        self.boundary_condition = boundary_condition
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequencies = None
        
        # Grid spacing
        # 0     h     2h    3h    ...  nh    L
        # x₀ --- x₁ --- x₂ --- x₃ --- ... --- xₙ₊₁
        # |_____|_____|_____|_____|_..._|_____|
        #  seg1  seg2  seg3  seg4  ...  seg(n+1)
        self.h = L / (n + 1)
        
        self.x = np.linspace(0, L, n+2)[1:-1]  # Remove boundary points
        self.y = np.linspace(0, L, n+2)[1:-1]
        
        self.build_matrix()
        
    def build_matrix(self):
        """Build the Laplacian matrix for the selected shape."""
        if self.shape == 'square':
            self.build_square_matrix()
        elif self.shape == 'rectangle':
            self.build_rectangle_matrix()
        elif self.shape == 'circle':
            self.build_circle_matrix()
            
    def build_square_matrix(self):
        """Build the Laplacian matrix for a square membrane."""
        n = self.n
        h = self.h
        c = self.c
        bc = self.boundary_condition
        
        # Laplacian matrix
        #     1
        #   1-4 1
        #     1        
        if self.use_sparse:
            from scipy.sparse import lil_matrix
            A = lil_matrix((n**2, n**2))
        else:
            A = np.zeros((n**2, n**2))
            
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                A[idx, idx] = -4
                if i > 0:
                    A[idx, idx - n] = 1
                if i < n - 1:
                    A[idx, idx + n] = 1
                if j > 0:
                    A[idx, idx - 1] = 1
                if j < n - 1:
                    A[idx, idx + 1] = 1
        self.A = A
        return A 
        
    def build_rectangle_matrix(self):
        """Build the Laplacian matrix for a rectangular membrane."""
        n = self.n
        h_x = self.L / (n + 1)
        h_y = (2 * self.L) / (n + 1) # Grid spacing in y direction (twice as long)
        
        # Laplacian matrix       
        if self.use_sparse:
            A = lil_matrix((n**2, n**2))
        else:
            A = np.zeros((n**2, n**2))
       
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                 # Diagonal element (adjusted for different spacing in x and y)
                A[idx, idx] = -2/h_x**2 - 2/h_y**2
                
                # Connect to neighbors
                if i > 0:  # left
                    A[idx, idx - n] = 1/h_x**2
                if i < n - 1:  # right
                    A[idx, idx + n] = 1/h_x**2
                if j > 0:  # down
                    A[idx, idx - 1] = 1/h_y**2
                if j < n - 1:  # up
                    A[idx, idx + 1] = 1/h_y**2
        self.A = A
        return A
                    
    def solve(self, num_modes=6):
        """
        Solve the eigenvalue problem to find eigenmodes and eigenfrequencies.
        
        Returns:
        eigenvalues: The eigenvalues (related to frequencies)
        eigenvectors: The eigenvectors (eigenmodes)
        """
        if not hasattr(self, 'A'):
            raise ValueError("Matrix not built yet. Call build_matrix() first.")
        
        A_scaled = self.A / (self.h**2)
        
        # Eigenvalue problem
        if self.use_sparse:
            eigenvalues, eigenvectors = sparse_linalg.eigsh(A_scaled, k=num_modes, which='SM')
        else:
            eigenvalues, eigenvectors = np.linalg.eig(A_scaled)
            
            # Sort eigenvalues and corresponding eigenvectors
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            eigenvalues = eigenvalues[:num_modes]
            
            # Ensure real-valued eigenvectors by taking the real part
            # and normalizing the eigenvectors
            eigenvectors = eigenvectors[:, :num_modes].real
            for i in range(num_modes):
                eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])
        
        # K = -lambda^2
        frequencies = np.sqrt(-eigenvalues)

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.frequencies = frequencies
        
        return eigenvalues, eigenvectors
        
    def plot_mode(self, mode_idx, figsize=(8, 6)):
        """Plot a specific eigenmode"""
        if not hasattr(self, 'eigenvectors'):
            raise ValueError("You need to call solve() before plotting")
            
        if self.shape == 'rectangle':
            # For rectangle, we need to consider the aspect ratio
            mode = self.eigenvectors[:, mode_idx].reshape(self.n, self.n)
            
            plt.figure(figsize=figsize)
            # Set extent to match the physical dimensions
            plt.imshow(mode, cmap='coolwarm', extent=[0, self.L, 0, 2*self.L])
        else:
            # For square
            mode = self.eigenvectors[:, mode_idx].reshape(self.n, self.n)
            
            plt.figure(figsize=figsize)
            plt.imshow(mode, cmap='coolwarm', extent=[0, self.L, 0, self.L])
        
        plt.colorbar(label='Amplitude')
        plt.title(f'Eigenmode {mode_idx+1}, Frequency: {self.frequencies[mode_idx]:.4f}')
        plt.xlabel('x', fontsize=labelsize)
        plt.ylabel('y', fontsize=labelsize)
        plt.tight_layout()
        
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