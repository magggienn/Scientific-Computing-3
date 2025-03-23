'''
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Contains the implementation for the leapfrog integration method.

Contains a class to initialise a leapfrog object. The object uses the leapfrog
method for 1d integration over t_max time steps. It also contains functionality
to include a driving force to the system to incur resonance. Lastly, the module
also includes plotting functions to show the evolution of the system and show
the result of applying a driving force.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


# Global settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
LABELSIZE = 35
TICKSIZE = 30


class LeapfrogHarmonicOscillator:
    def __init__(self, k, m=1.0, dt=0.01, t_max=10.0, F_ext=None):
        """
        Initializes the Leapfrog Harmonic Oscillator.

        Args:
            k: Spring constant (float)
            m: Mass of the oscillator (float, default=1.0)
            dt: Time step for integration (float, default=0.01)
            t_max: Maximum simulation time (float, default=10.0)
            F_ext: External force function, dependent on time (callable, default=None)
        """
        self.k = k          # Spring constant
        self.m = m          # Mass
        self.dt = dt        # Time step
        self.t_max = t_max  # Maximum simulation time
        self.F_ext = F_ext  # External driving force function

    def force(self, x, t):
        """
        Applies the force on the system according to Hooke's law.

        Args:
            x: the position (float)
            t: the time at which the system is (float)

        Returns:
            total force on the system at time t and position x (float)
        """
        F_harmonic = -self.k * x  # Hooke's Law
        F_drive = self.F_ext(t) if self.F_ext else 0  # External force
        return F_harmonic + F_drive

    def integrate(self, x0, v0):
        """
        Integrates the equations of motion using the Leapfrog method.

        Args:
            x0: Initial position (float)
            v0: Initial velocity (float)

        Returns:
            t_vals: Time values (numpy array)
            x_vals: Position values (numpy array)
            v_vals: Velocity values (numpy array)
        """
        num_steps = int(self.t_max / self.dt)
        t_vals = np.linspace(0, self.t_max, num_steps)
        x_vals = np.zeros(num_steps)
        v_vals = np.zeros(num_steps)

        x_vals[0] = x0

        # Initial half-step velocity
        v_half = v0 + 0.5 * self.dt * self.force(x0, 0) / self.m

        for i in range(1, num_steps):
            t = t_vals[i-1]
            x_vals[i] = x_vals[i-1] + self.dt * v_half
            v_half += self.dt * self.force(x_vals[i], t + 0.5 * self.dt) / self.m

            # Approximate v at full step
            v_vals[i] = v_half - 0.5 * self.dt * self.force(x_vals[i], t) / self.m

        return t_vals, x_vals, v_vals

    def plot_results(self, x0, v0, ks):
        """
        Plots position and velocity over time for different values of k.

        Args:
            x0: Initial position (float)
            v0: Initial velocity (float)
            ks: List of spring constants to compare (list of floats)
        """
        factor = 1.5

        plt.figure(figsize=(10, 10))

        # Position vs Time
        plt.subplot(2, 1, 1)  # Stack plots vertically
        for i, k in enumerate(ks):
            self.k = k
            t_vals, x_vals, v_vals = self.integrate(x0, v0)
            plt.plot(t_vals, x_vals, label=f'$k={k}$', linewidth=2)

        plt.title(r"Position vs Time", fontsize=LABELSIZE * factor)
        plt.xlabel(r"Time", fontsize=LABELSIZE * factor)
        plt.ylabel(r"Position", fontsize=LABELSIZE * factor)
        plt.xticks(fontsize=TICKSIZE * factor)
        plt.yticks(fontsize=TICKSIZE * factor)
        plt.legend(fontsize=(TICKSIZE - 2) * factor)

        # Velocity vs Time
        plt.subplot(2, 1, 2)
        for i, k in enumerate(ks):
            self.k = k
            t_vals, x_vals, v_vals = self.integrate(x0, v0)
            plt.plot(t_vals, v_vals, label=f'$k={k}$', linewidth=2)

        plt.title(r"Velocity vs Time", fontsize=LABELSIZE * factor)
        plt.xlabel(r"Time", fontsize=LABELSIZE * factor)
        plt.ylabel(r"Velocity", fontsize=LABELSIZE * factor)
        plt.xticks(fontsize=TICKSIZE * factor)
        plt.yticks(fontsize=TICKSIZE * factor)
        plt.legend(fontsize=(TICKSIZE - 2) * factor)

        plt.tight_layout()
        plt.show()

    def plot_phase_space(self, x0, v0, frequencies, rf):
        """
        Plots the phase space trajectory for different driving frequencies with arrows to indicate the direction.

        Args:
            x0: Initial position (float)
            v0: Initial velocity (float)
            frequencies: List of driving frequencies to compare (list of floats)
        """
        plt.figure(figsize=(12, 10))

        for f in frequencies:
            self.F_ext = lambda t: np.sin(2 * np.pi * f * t)  # Sinusoidal driving force
            _, x_vals, v_vals = self.integrate(x0, v0)

            # Plot the phase space trajectory with arrowheads showing the direction
            line, = plt.plot(x_vals, v_vals, label=f'f={round(f / rf, 2)} * ' + r"$f_{res}$", linewidth=3, marker='o', markersize=3)

            line_color = line.get_color()

            # Add arrows along the line at regular intervals
            for i in range(0, len(x_vals) - 1, len(x_vals) // 10):  # Add arrows at intervals
                plt.arrow(x_vals[i], v_vals[i], x_vals[i+1] - x_vals[i], v_vals[i+1] - v_vals[i],
                        head_width=0.15, head_length=0.12, fc=line_color, ec=line_color)

        # Configure labels, title, and legend with global font settings
        plt.xlabel(r"Position", fontsize=LABELSIZE)
        plt.ylabel(r"Velocity", fontsize=LABELSIZE)
        plt.title(r"Phase Space Plot", fontsize=LABELSIZE)
        plt.xticks(fontsize=TICKSIZE)
        plt.yticks(fontsize=TICKSIZE)
        plt.legend(fontsize=TICKSIZE - 2)

        plt.show()


def harmonic_oscillator(t, y, k, m, F_ext):
    """
    Defines the system of equations for a damped harmonic oscillator.

    Args:
        t: Time variable (float)
        y: State vector [position, velocity] (list of floats)
        k: Spring constant (float)
        m: Mass of the oscillator (float)
        F_ext: External force function (callable, default=None)

    Returns:
        List containing the derivatives [dx/dt, dv/dt]
    """
    x, v = y
    dxdt = v
    dvdt = (-k * x + (F_ext(t) if F_ext else 0)) / m
    return [dxdt, dvdt]


def compare_energy_conservation(x0, v0, k, dt, t_max, F_ext=None):
    """
    Compares energy conservation using the Leapfrog method and RK45.

    Args:
        x0: Initial position (float)
        v0: Initial velocity (float)
        k: Spring constant (float)
        dt: Time step for Leapfrog method (float)
        t_max: Maximum simulation time (float)
        F_ext: External force function (callable, default=None)
    """
    lf_oscillator = LeapfrogHarmonicOscillator(k, dt=dt, t_max=t_max, F_ext=F_ext)
    t_vals, x_vals, v_vals = lf_oscillator.integrate(x0, v0)
    energy_leapfrog = 0.5 * k * x_vals**2 + 0.5 * v_vals**2

    sol = solve_ivp(harmonic_oscillator, [0, t_max], [x0, v0], args=(k, 1, F_ext),
                    t_eval=t_vals, method='RK45')
    energy_rk45 = 0.5 * k * sol.y[0]**2 + 0.5 * sol.y[1]**2

    plt.figure(figsize=(12, 8))
    plt.plot(t_vals, energy_leapfrog, label=r'Leapfrog', linewidth=2)
    plt.plot(t_vals, energy_rk45, label=r'RK45', linewidth=2)

    # Configure labels, title, and legend with global font settings
    plt.xlabel(r"Time", fontsize=LABELSIZE)
    plt.ylabel(r"Total Energy", fontsize=LABELSIZE)
    plt.title(r"Energy Conservation Comparison", fontsize=LABELSIZE)
    plt.xticks(fontsize=TICKSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.legend(fontsize=TICKSIZE - 2)

    plt.show()


def calculate_natural_frequency(k, m=1.0):
    """Calculates the natural frequency of the oscillator."""
    return (1 / (2 * np.pi)) * np.sqrt(k / m)


def compare_energy_growth(x0, v0, k, dt, t_max, f_drive):
    F_ext = lambda t: np.sin(2 * np.pi * f_drive * t)  # Driving force

    # Leapfrog simulation
    lf_oscillator = LeapfrogHarmonicOscillator(k, dt=dt, t_max=t_max, F_ext=F_ext)
    t_vals, x_vals, v_vals = lf_oscillator.integrate(x0, v0)
    energy_leapfrog = 0.5 * k * x_vals**2 + 0.5 * v_vals**2

    # RK45 simulation
    sol = solve_ivp(harmonic_oscillator, [0, t_max], [x0, v0], args=(k, 1, F_ext),
                    t_eval=t_vals, method='RK45')
    energy_rk45 = 0.5 * k * sol.y[0]**2 + 0.5 * sol.y[1]**2

    # Plot energy growth
    plt.figure(figsize=(8, 5))
    plt.plot(t_vals, energy_leapfrog, label='Leapfrog', linestyle='--')
    plt.plot(t_vals, energy_rk45, label='RK45')
    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.title(f"Energy Growth Comparison (f_drive={f_drive})")
    plt.legend()
    plt.show()


def main():

    def p1():
        k = 1.0
        dt = 0.001
        t_max = 20
        lf_oscillator = LeapfrogHarmonicOscillator(k=k, dt=dt, t_max=t_max)
        # lf_oscillator.plot_results(x0=1.0, v0=0.0, ks=[0.5, 1.0, 2.0])
        compare_energy_conservation(x0=1.0, v0=0.0, k=1.0, dt=0.01, t_max=50)

    def p2():
        # reinitialisation with driving force
        k = 1.0
        dt = 0.001
        t_max = 10
        lf_oscillator = LeapfrogHarmonicOscillator(k=k, dt=dt, t_max=t_max)

        cf = calculate_natural_frequency(k)

        # Phase space plot with driving force close to the natural frequency of the oscillator
        lf_oscillator.plot_phase_space(x0=1.0, v0=0.0,
                                       frequencies=[.4 * cf, cf, 1.6 * cf], rf=cf)

        k = 1.0
        dt = 0.001
        t_max = 50
        lf_oscillator = LeapfrogHarmonicOscillator(k=k, dt=dt, t_max=t_max)
        lf_oscillator.plot_phase_space(x0=1.0, v0=0.0,
                                       frequencies=[cf], rf=cf)


    p2()


if __name__ == "__main__":
    main()
