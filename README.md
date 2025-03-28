# Scientific Computing Assignment 3

## Membrane Eigenmodes, Direct Methods, and Leapfrog Integration

This project contains implementations of numerical methods for solving
eigenvalue problems related to vibrating membranes, steady-state diffusion
equations, and efficient time integration using the leapfrog method. The report
explores three different membrane shapes and their eigenmodes, analyzes the
direct methods for diffusion problems, and compares the leapfrog method's
efficiency and energy conservation in harmonic oscillators.

## Usage

To run the main script, which generates all the figures used in the report, use the following command:

```bash
python src/main.py
```

For more information on what to run in main, see main function in main.py


## src Folder Structure

Here’s a summary of all the files in the `src` folder:

### `main.py`
The main script that orchestrates the execution of different sections of the project. It allows you to:
- Solve the membrane eigenvalue problem (`solve_membrane()`).
- Run the direct method for solving the diffusion equation (`solve_direct_method()`).
- Plot results from the leapfrog method for a harmonic oscillator (`plot_leapfrog()`).

### `membrane_solver.py`
Contains the `MembraneSolver` class that solves the eigenvalue problem for various membrane shapes. It handles:
- Calculation of eigenmodes and eigenfrequencies.
- Sparse and dense matrix solver comparisons.
- Visualization of the results, including plotting and animating the eigenmodes.

### `direct_method.py`
Contains the `SolveDirectMethod` class that solves the steady-state concentration problem for diffusion using direct methods. It:
- Discretizes the diffusion equation.
- Constructs the necessary matrices for solving the system.
- Visualizes the results.

### `leapfrog.py`
Contains the `LeapfrogHarmonicOscillator` class and related functions. It implements:
- The leapfrog method for solving a simple one-dimensional harmonic oscillator.
- Functions for energy conservation comparison between different methods.
- Phase space comparison of oscillators with varying frequencies.

### `plot_functions.py`
Contains plotting functions for visualizing results:
- `plot_eigenfrequencies_vs_L`: Plots eigenfrequencies as a function of size \( L \) for different shapes.
- `plot_combined_performance`: Combines performance metrics for both sparse and dense solvers and plots the comparison.

### `plot_matrix_M.py`
Contains functions to visualize the structure of matrices:
- `visualize_all_matrices_in_one_row`: Visualizes multiple matrices side by side.
- `visualize_matrix_structure`: Provides a more detailed view of the matrix structure for better understanding.

Each of these files contains relevant code and functions to handle specific tasks and can be executed as described in `main.py` for generating results and plots.


## figures Folder Structure

The `figures` folder contains subfolders with plots generated from the code:

### `boundaries/`
Contains figures related to the boundary conditions of the membrane problems, visualizing fixed boundaries for different shapes.

### `dense/`
Stores figures comparing the performance and results using dense matrix solvers for membrane eigenvalue problems.

### `frog/`
Includes figures from the leapfrog method, such as position/velocity plots and energy conservation comparisons for harmonic oscillators.

### `performance/`
Holds figures showing the performance comparison between sparse and dense solvers for the membrane eigenvalue problem.

### `sparse/`
Contains figures comparing the results and performance using sparse matrix solvers for membrane eigenvalue problems.

These subfolders organize the figures based on the type of analysis or method used in the project.


## animation Folder Structure

The `animation` folder contains animations of eigenmodes for different membrane shapes:

### `circle/`
Contains animations for the first six eigenmodes of a circular membrane. Each mode corresponds to a different vibration pattern of the membrane.

### `rectangle/`
Contains animations for the first six eigenmodes of a rectangular membrane, visualizing the vibration patterns for each mode.

### `square/`
Contains animations for the first six eigenmodes of a square membrane, showing the membrane's deformation for each eigenmode.

These subfolders include animations for modes 1-6, providing a visual representation of the eigenmodes for each shape.

