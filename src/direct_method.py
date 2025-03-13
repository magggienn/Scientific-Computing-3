import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import diags
import matplotlib.pyplot as plt

class SolveDirectMethod:
    '''
    Class that solves the diffusion equation using direct method
    '''
    def __init__(self, Nx = 100, Ny=100):
        self.radius = 2.0
        self.Nx = Nx
        self.Ny = Ny
        self.source_loc = [0.6, 1.2]
        self.x = np.linspace(-self.radius, self.radius, Nx)
        self.y = np.linspace(-self.radius, self.radius, Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Mask inner circle
        self.mask = np.array([[xi**2 + yi**2 <= self.radius **2 for xi in self.x] for yi in self.y])
        self.num_points = np.sum(self.mask)
        self.indices = np.full((Ny, Nx), -1, dtype=int)
        self.indices[self.mask] = np.arange(self.num_points)

    def source_calc(self, x, y):
        '''
        Source term function
        '''
        return 1 if np.isclose(x, self.source_loc[0], atol = self.dx) and np.isclose(y, self.source_loc[1], atol = self.dy) else 0
    
    def build_M_matrix(self):
        '''
        Construct M matrix from Laplacian==> makes sparse matrix from diagonals
        '''
        print("Start building M matrix")
        #finite difference discretization laplacian (5 point)
        main_diag = np.ones(self.num_points) * -4 
        horizontal_diag = np.ones(self.num_points - 1) # horizontal neighbors
        horizontal_diag[np.arange(1, self.num_points) % self.Nx == 0] = 0 # remove connections that wrap around
        vertical_diag = np.ones(self.num_points - self.Nx) # vertical neighbors

        # Set diagonals
        diagonals = [main_diag, horizontal_diag, horizontal_diag, vertical_diag, vertical_diag]
        offsets = [0, -1, 1, -self.Nx, self.Nx]

        return diags(diagonals, offsets, shape=(self.num_points, self.num_points), format='csr')
    
    def build_b_vector(self):
        '''
        Build b vector
        '''
        b_vec = np.zeros(self.num_points)
        # Reset the b vector first
        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.mask[i, j]:
                    row = int(self.indices[i, j])
                    x_pos = self.x[j]
                    y_pos = self.y[i]
                    b_vec[row] = self.source_calc(x_pos, y_pos)
        return b_vec
    
    def solve(self):
        '''
        Solve the diffusion equation using direct method
        '''
        M = self.build_M_matrix()
        b = self.build_b_vector()
        self.c = scipy.sparse.linalg.spsolve(M, b)
        self.result = np.zeros((self.Nx, self.Ny))
        self.result[self.mask] = self.c

    def plot(self):
        '''
        Plot the solution
        '''
        plt.figure(figsize=(6, 6))
        plt.contourf(self.x, self.y, self.result, levels=50, cmap='jet')
        plt.colorbar(label='Concentration')
        plt.title('Steady-State Concentration Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('../figures/direct_method.pdf')
        plt.show()

        return
if __name__ == "__main__":
    diffusion = SolveDirectMethod()
    diffusion.solve()  
    diffusion.plot()