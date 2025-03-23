import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LABELSIZE =25
TICKSIZE = 23

class SolveDirectMethod:
    '''
    Class that solves the diffusion equation using the direct method
    '''
    def __init__(self, Nx=100, Ny=100):
        self.radius = 2.0
        self.Nx = Nx
        self.Ny = Ny
        self.source_loc = (0.6, 1.2)
        self.x = np.linspace(-self.radius, self.radius, Nx)
        self.y = np.linspace(-self.radius, self.radius, Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Mask to indicate the circle region
        self.mask = np.array([[xi**2 + yi**2 <= self.radius**2 for xi in self.x] for yi in self.y])
        self.num_points = np.sum(self.mask)
        self.indices = np.full((Ny, Nx), -1, dtype=int)
        self.indices[self.mask] = np.arange(self.num_points)

    def source_calc(self, x, y):
        '''
        Assigns the source at the closest grid point
        '''
        closest_x = min(self.x, key=lambda xi: abs(xi - self.source_loc[0]))
        closest_y = min(self.y, key=lambda yi: abs(yi - self.source_loc[1]))
        return 1 if (x == closest_x and y == closest_y) else 0
    
    def build_M_matrix(self):
        '''
        Constructs M matrix from the finite difference discretization of the Laplacian
        '''
        laplace_main = []
        laplace_i = []
        laplace_j = []

        for i in range(self.Ny):
            for j in range(self.Nx):
                if self.mask[i, j]: 
                    idx = self.indices[i, j]

                    laplace_main.append(-4)
                    laplace_i.append(idx)
                    laplace_j.append(idx)

                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.Ny and 0 <= nj < self.Nx and self.mask[ni, nj]:
                            neighbor_idx = self.indices[ni, nj]
                            laplace_main.append(1)
                            laplace_i.append(idx)
                            laplace_j.append(neighbor_idx)

        M = scipy.sparse.csr_matrix((laplace_main, (laplace_i, laplace_j)), shape=(self.num_points, self.num_points))
        return M
    
    def build_b_vector(self):
        '''
        Constructs the b vector incorporating the source term
        '''
        b_vec = np.zeros(self.num_points)
        for i in range(self.Ny):
            for j in range(self.Nx):
                if self.mask[i, j]:
                    row = self.indices[i, j]
                    x_pos = self.x[j]
                    y_pos = self.y[i]
                    b_vec[row] = self.source_calc(x_pos, y_pos)
        return b_vec
    
    def solve(self):
        '''
        Solves the diffusion equation using a direct method
        '''
        M = self.build_M_matrix()
        b = self.build_b_vector()
        self.c = scipy.sparse.linalg.spsolve(M, -b)
        self.result = np.zeros((self.Nx, self.Ny))
        self.result[self.mask] = self.c

    def plot(self, filename='figures/direct_method.pdf'):
        '''
        Plots the solution.
        '''
        fig, ax = plt.subplots(figsize=(6.5, 6)) 
        contour = ax.contourf(self.x, self.y, self.result, levels=100, cmap='Spectral')

        # Circle boundary
        circle = plt.Circle((0, 0), self.radius, fill=False, color='black', linestyle='-', linewidth=0.5)
        ax.add_patch(circle)

        ax.set_xlabel('x', fontsize=LABELSIZE)
        ax.set_ylabel('y', fontsize=LABELSIZE)
        ax.set_aspect('equal') 
        ax.tick_params(labelsize=TICKSIZE)

        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Concentration', fontsize=LABELSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)

        # To make sure the tick values aren't too many 
        min_val = np.min(self.result)
        max_val = np.max(self.result)
        cbar.set_ticks(np.linspace(min_val, max_val, 6))  # e.g. 6 steps
        cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(min_val, max_val, 6)])
        
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    diffusion = SolveDirectMethod()
    diffusion.solve()
    diffusion.plot()
