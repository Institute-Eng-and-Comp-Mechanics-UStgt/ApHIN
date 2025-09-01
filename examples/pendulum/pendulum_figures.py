import os
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis

# Constants
N_STATES = 2
N_TRAJECTORIES = 6
TRAJECTORY_INDEX = 3

# Define a named tuple for experiment properties
Experiment = namedtuple("Experiment", ["file_path", "linestyle", "color"])


def load_experiment_results(result_dir, experiments):
    """Load results for all experiments."""
    results = {}
    for key, experiment in experiments.items():
        file_dir = os.path.join(result_dir, "..", experiment.file_path)
        if not os.path.exists(file_dir):
            raise FileNotFoundError(
                f"Results file for experiment '{key}' not found at {file_dir}"
            )
        data = np.genfromtxt(file_dir, delimiter=",", skip_header=1)
        t = data[:, 0]
        x = (
            data[:, 1 : N_STATES * N_TRAJECTORIES + 1]
            .reshape(1000, N_TRAJECTORIES, N_STATES)
            .transpose([1, 0, 2])
        )
        dxdt = (
            data[:, N_STATES * N_TRAJECTORIES + 1 : (2 * N_STATES * N_TRAJECTORIES + 1)]
            .reshape(1000, N_TRAJECTORIES, N_STATES)
            .transpose([1, 0, 2])
        )
        dxddt = (
            data[:, (2 * N_STATES * N_TRAJECTORIES + 1) :]
            .reshape(1000, N_TRAJECTORIES, N_STATES)
            .transpose([1, 0, 2])
        )
        results[key] = dict(t=t, x=x, dxdt=dxdt, dxddt=dxddt)
    return results


def plot_trajectories(results, experiments):
    """Plot state, velocity, and acceleration trajectories."""
    fig, axs = plt.subplots(3, N_STATES)
    for key, result in results.items():
        experiment = experiments[key]
        for state_index, state_name in enumerate(["x", "y"]):
            axs[0, state_index].plot(
                result["t"],
                result["x"][TRAJECTORY_INDEX, :, state_index],
                linestyle=experiment.linestyle,
                color=experiment.color,
            )
            axs[1, state_index].plot(
                result["t"],
                result["dxdt"][TRAJECTORY_INDEX, :, state_index],
                linestyle=experiment.linestyle,
                color=experiment.color,
            )
            axs[2, state_index].plot(
                result["t"],
                result["dxddt"][TRAJECTORY_INDEX, :, state_index],
                linestyle=experiment.linestyle,
                color=experiment.color,
            )
            axs[0, state_index].set_ylabel(f"${{\\rho}}_{state_name}$")
            axs[1, state_index].set_ylabel(f"$\\dot{{\\rho}}_{state_name}$")
            axs[2, state_index].set_xlabel("time in s")
            axs[2, state_index].set_ylabel(f"$\\ddot{{\\rho}}_{state_name}$")
    fig.legend(
        list(experiments.keys()), loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    plt.savefig("pendulum_figures.png")


def main():
    aphin_vis.setup_matplotlib(save_plots=False)
    working_dir = os.path.dirname(__file__)
    configuration = Configuration(working_dir)
    pd_cfg = configuration.cfg_dict
    _, _, _, result_dir = configuration.directories

    # Refactored experiments dictionary using namedtuple
    experiments = {
        "reference": Experiment(
            file_path="pendulum_model_aphin_linear/reference.csv",
            linestyle="solid",
            color="black",
        ),
        "phin": Experiment(
            file_path="pendulum_model_phin/phin.csv", linestyle="dashdot", color="cyan"
        ),
        "aphin_linear": Experiment(
            file_path="pendulum_model_aphin_linear/aphin_linear.csv",
            linestyle="dotted",
            color="purple",
        ),
        "aphin_nonlinear": Experiment(
            file_path="pendulum_model_aphin_nonlinear/aphin_nonlinear.csv",
            linestyle="dashed",
            color="magenta",
        ),
    }

    results = load_experiment_results(result_dir, experiments)
    plot_trajectories(results, experiments)


if __name__ == "__main__":
    main()
