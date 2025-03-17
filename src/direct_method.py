import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        Plots the solution
        '''
        plt.figure(figsize=(6, 6))
        plt.contourf(self.x, self.y, self.result, levels=50, cmap='Spectral')
        plt.colorbar(label='Concentration')
        plt.title('Steady-State Concentration Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(filename)
        plt.show()

    def animate(self, duration=10, fps=30, filename="animations/AnimateDirectMethod.mkv"):
        '''
        Animate the solution
        '''
        if not hasattr(self, 'result'):
            raise ValueError("You need to call solve() before animating.")

        # Calculate total number of frames
        num_frames = duration * fps

        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=(6, 6))
        max_result = np.max(self.result)
        if max_result <= 0:
            raise ValueError("Maximum result value must be positive to create contour levels.")
        
        levels = np.linspace(0, max_result, 50)
        contour = ax.contourf(self.x, self.y, np.zeros_like(self.result), levels=levels, cmap='jet')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Concentration')
        ax.set_title('Steady-State Concentration Distribution (Animation)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        def update(frame):
            """Update function for the animation."""
            progress = (frame + 1) / num_frames
            current_result = self.result * progress
            ax.clear()
            contour = ax.contourf(self.x, self.y, current_result, levels=levels, cmap='jet')
            ax.set_title(f'Steady-State Concentration Distribution (iter={progress:.2f}s)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            return contour.collections

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=True)

        if filename:
            anim.save(filename, writer='ffmpeg', fps=fps)
            print(f"Animation saved to {filename}")

        plt.show()

if __name__ == "__main__":
    diffusion = SolveDirectMethod()
    diffusion.solve()
    diffusion.plot()
