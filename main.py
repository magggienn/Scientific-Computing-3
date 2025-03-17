from src.membrane_solver import MembraneSolver
import matplotlib.pyplot as plt
from src.direct_method import SolveDirectMethod


def solve_membrane():
        # square membrane
    square = MembraneSolver(shape='square', L=1.0, n=30)
    square.solve(num_modes=6)
    square.plot_mode(0)
    plt.savefig(f'figures/square_modes.pdf')
    
    # rectangular membrane
    rectangle = MembraneSolver(shape='rectangle', L=1.0, n=30)
    rectangle.solve(num_modes=6)
    rectangle.plot_mode(0)
    plt.savefig(f'figures/rectangle_modes.pdf')
    # anim = square.animate_mode(mode_idx=0, duration=5, filename="square_mode1_vibration.mp4")
    # anim = rectangle.animate_mode(mode_idx=0, duration=5, filename="rectangle_mode1_vibration.mp4")

def solve_direct_method():
    diffusion = SolveDirectMethod()
    diffusion.solve()
    diffusion.animate()
    diffusion.plot()

if __name__ == "__main__":
    #solve_membrane()
    solve_direct_method()
