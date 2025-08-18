import os
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis

# Constants
N_STATES = 2
N_TRAJECTORIES = 6
TRAJECTORY_INDEX = 2

# Define a named tuple for experiment properties
Experiment = namedtuple("Experiment", ["file_path", "linestyle", "color", "marker"])

def load_experiment_results(result_dir, experiments):
    """Load results for all experiments."""
    results = {}
    for key, experiment in experiments.items():
        file_dir = os.path.join(result_dir, experiment.file_path)
        if not os.path.exists(file_dir):
            raise FileNotFoundError(f"Results file for experiment '{key}' not found at {file_dir}")
        eigenvalues = np.genfromtxt(os.path.join(file_dir, "eigenvalues_phin.csv"), delimiter=',', skip_header=1)
        t = data[:, 0]
        x = data[:, 1:N_STATES * N_TRAJECTORIES + 1].reshape(1000, N_TRAJECTORIES, N_STATES).transpose([1, 0, 2])
        dxdt = data[:, N_STATES * N_TRAJECTORIES + 1:(2 * N_STATES * N_TRAJECTORIES + 1)].reshape(1000, N_TRAJECTORIES, N_STATES).transpose([1, 0, 2])
        dxddt = data[:, (2 * N_STATES * N_TRAJECTORIES + 1):].reshape(1000, N_TRAJECTORIES, N_STATES).transpose([1, 0, 2])
        results[key] = dict(t=t, x=x, dxdt=dxdt, dxddt=dxddt)
    return results

def plot_trajectories(results, experiments):
    """Plot state, velocity, and acceleration trajectories."""
    fig, axs = plt.subplots(3, N_STATES)
    for key, result in results.items():
        experiment = experiments[key]
        for state_index, state_name in enumerate(["x", "y"]):
            axs[0, state_index].plot(result["t"], result["x"][TRAJECTORY_INDEX, :, state_index],
                                     linestyle=experiment.linestyle, color=experiment.color)
            axs[1, state_index].plot(result["t"], result["dxdt"][TRAJECTORY_INDEX, :, state_index],
                                     linestyle=experiment.linestyle, color=experiment.color)
            axs[2, state_index].plot(result["t"], result["dxddt"][TRAJECTORY_INDEX, :, state_index],
                                     linestyle=experiment.linestyle, color=experiment.color)
            axs[0, state_index].set_ylabel(f"${{\rho}}_{state_name}$")
            axs[1, state_index].set_ylabel(f"$\\dot{{\\rho}}_{state_name}$")
            axs[2, state_index].set_xlabel("time in s")
            axs[2, state_index].set_ylabel(f"$\\ddot{{\\rho}}_{state_name}$")
    fig.legend(list(experiments.keys()), loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def main():
    aphin_vis.setup_matplotlib(save_plots=False)
    working_dir = os.path.dirname(__file__)
    configuration = Configuration(working_dir)
    _, _, _, result_dir = configuration.directories

    # Refactored experiments dictionary using namedtuple
    experiments = {
        "reference": Experiment(file_path="phin", linestyle="solid", color="black", marker="o"),
        "lti": Experiment(file_path="lti", linestyle="dashdot", color="cyan", marker="triangle"),
        "mi": Experiment(file_path="mi", linestyle="dotted", color="magenta", marker="square"),
        "phin": Experiment(file_path="phin", linestyle="dashed", color="orange", marker="diamond"),
    }

    results = load_experiment_results(result_dir, experiments)
    plot_trajectories(results, experiments)

if __name__ == "__main__":
    main()