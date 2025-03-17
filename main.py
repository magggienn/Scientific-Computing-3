from src.membrane_solver import MembraneSolver
import matplotlib.pyplot as plt
import numpy as np
from src.direct_method import SolveDirectMethod


def solve_membrane():
    # square membrane
    square = MembraneSolver(c=1,shape='square', L=1.0, n=30, use_sparse=True)
    square.solve(num_modes=5)
    eigenvalues, _ = square.solve(num_modes=5)
    # In your solve method, print a few eigenvalues to check:
    print("First few eigenvalues:", eigenvalues[:5])

    # For a square membrane with side length L=1, the analytical eigenvalues are:
    # λ = -π²(m² + n²) where m, n are positive integers
    analytical = [-(np.pi**2)*(m**2 + n**2) for m in range(1, 3) for n in range(1, 3)]
    analytical.sort()
    print("Analytical eigenvalues:", analytical[:5])
    print("Eigenvalues:", eigenvalues)


    square.plot_mode(0)
    plt.savefig(f'figures/square_modes.pdf')
    
    # # rectangular membrane
    # rectangle = MembraneSolver(shape='rectangle', L=1.0, n=30)
    # rectangle.solve(num_modes=10)
    # rectangle.plot_mode(0)
    # plt.savefig(f'figures/rectangle_modes.pdf')
    # anim = square.animate_mode(mode_idx=0, duration=5, filename="square_mode1_vibration.mp4")
    # anim = rectangle.animate_mode(mode_idx=0, duration=5, filename="rectangle_mode1_vibration.mp4")
    
    # circular membrane
    # circle = MembraneSolver(shape='circle', L=1.0, n=6)
    # circle.solve(num_modes=6)
    # circle.plot_mode(0)
    # plt.savefig(f'figures/circle_modes.pdf')
    # anim = circle.animate_mode(mode_idx=0, duration=5, filename="circle_mode1_vibration.mp4")
    

def solve_direct_method():
    diffusion = SolveDirectMethod()
    diffusion.solve()
    diffusion.animate()
    diffusion.plot()

if __name__ == "__main__":
    solve_membrane()
    # solve_direct_method()
