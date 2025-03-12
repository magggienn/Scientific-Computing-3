from src.membrane_solver import MembraneSolver
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # square membrane
    square = MembraneSolver(shape='square', L=1.0, n=30)
    square.solve(num_modes=6)
    square.plot_mode(0)
    plt.savefig(f'figures/square_modes.png')
    
    # rectangular membrane
    rectangle = MembraneSolver(shape='rectangle', L=1.0, n=30)
    rectangle.solve(num_modes=6)
    rectangle.plot_mode(0)
    plt.savefig(f'figures/rectangle_modes.png')
    # anim = square.animate_mode(mode_idx=0, duration=5, filename="square_mode1_vibration.mp4")
    # anim = rectangle.animate_mode(mode_idx=0, duration=5, filename="rectangle_mode1_vibration.mp4")
