import os
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis
from matplotlib.lines import Line2D

# Constants
N_STATES = 3
N_T = 2000
N_TRAJECTORIES = 60
TRAJECTORY_INDEX = 3

# Define a named tuple for experiment properties
Experiment = namedtuple("Experiment", ["file_path", "linestyle", "color", "marker"])


def load_experiment_results(result_dir, experiments):
    """Load results for all experiments."""
    results = {}
    for key, experiment in experiments.items():
        file_dir = os.path.join(result_dir, experiment.file_path)
        if not os.path.exists(file_dir):
            raise FileNotFoundError(
                f"Results file for experiment '{key}' not found at {file_dir}"
            )
        # eigenvalues
        eigenvalues = (
            np.genfromtxt(
                os.path.join(file_dir, "eigenvalues.csv"), delimiter=",", skip_header=1
            )
            .reshape(-1, 2, N_TRAJECTORIES)
            .transpose([2, 0, 1])
        )
        # state trajectories
        data = np.empty((N_T, N_TRAJECTORIES * N_STATES + 1, 0))
        for i in range(N_STATES):
            data = np.concatenate(
                [
                    data,
                    np.genfromtxt(
                        os.path.join(file_dir, f"state_{i}_trajectories.csv"),
                        delimiter=",",
                        skip_header=1,
                    )[:, :, np.newaxis],
                ],
                axis=-1,
            )

        t = data[:, 0, 0]
        x = data[:, 1 : N_TRAJECTORIES + 1].transpose([1, 0, 2])

        # state errors
        if key == "reference":
            t_error, state_error = None, None
        else:
            data = np.genfromtxt(
                os.path.join(file_dir, "rms_error_state_dom0.csv"),
                delimiter=",",
                skip_header=1,
            ).transpose([1, 0])
            t_error = data[0]
        state_error = data[1 : N_TRAJECTORIES + 1]
        matrix_plot_dir = (
            os.path.join(file_dir, "compare_matrices.png")
            if os.path.exists(os.path.join(file_dir, "compare_matrices.png"))
            else None
        )
        results[key] = dict(
            t=t,
            x=x,
            t_error=t_error,
            state_error=state_error,
            eigenvalues=eigenvalues,
            matrix_plot=matrix_plot_dir,
        )
    return results


def plot_trajectories(results, experiments):
    """Plot state, velocity, and acceleration trajectories."""
    msd_dir = os.path.dirname(__file__)
    # %% show matrix plot
    for key, result in results.items():
        if result["matrix_plot"] is not None and key != "reference":
            fig, ax = plt.subplots(figsize=(6, 6))
            img = plt.imread(result["matrix_plot"])
            ax.imshow(img)
            ax.set_title(f"Matrix comparison for {key}")
            ax.axis("off")
            plt.tight_layout()
            plt.show(block=False)
            plt.savefig(os.path.join(msd_dir, f"msd_traj_{key}.png"))

    # %% Eigenvalue plot
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for key, result in results.items():
        experiment = experiments[key]
        # Eigenvalue plot
        eigenvalues = result["eigenvalues"]
        axs[0].scatter(
            eigenvalues[TRAJECTORY_INDEX, :, 0],
            eigenvalues[TRAJECTORY_INDEX, :, 1],
            color=experiment.color,
            marker=experiment.marker,
            label=key,
        )
        # state trajectory plot
        axs[1].plot(
            result["t"],
            result["x"][TRAJECTORY_INDEX, :, :],
            linestyle=experiment.linestyle,
            color=experiment.color,
            label=key,
        )
        # error plot
        if result["t_error"] is not None:
            for i in range(N_TRAJECTORIES):
                axs[2].plot(
                    result["t_error"],
                    result["state_error"][i, :],
                    linestyle=experiment.linestyle,
                    alpha=0.03,
                    color=experiment.color,
                    label="_nolegend_",
                )
            axs[2].plot(
                result["t_error"],
                result["state_error"].mean(axis=0),
                linestyle=experiment.linestyle,
                color=experiment.color,
                label="_nolegend_",
                zorder=10,
            )

    axs[0].set_xlabel("$Re(\lambda)$")
    axs[0].set_ylabel("$Im(\lambda)$")
    axs[1].set_xlabel("time in s")
    axs[1].set_ylabel(r"$\bm{q}$")
    axs[2].set_xlabel("time in s")
    axs[2].set_ylabel("rel RMS error")

    # Create custom legend
    custom_lines = [
        Line2D(
            [0],
            [0],
            color=experiments["reference"].color,
            linestyle=experiments["reference"].linestyle,
            label="Reference",
        ),
        Line2D(
            [0],
            [0],
            color=experiments["lti"].color,
            linestyle=experiments["lti"].linestyle,
            label="LTI",
        ),
        Line2D(
            [0],
            [0],
            color=experiments["phin"].color,
            linestyle=experiments["phin"].linestyle,
            label="PHIN",
        ),
        Line2D(
            [0],
            [0],
            color=experiments["mi"].color,
            linestyle=experiments["mi"].linestyle,
            label="MI",
        ),
    ]
    fig.legend(
        handles=custom_lines, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=4
    )

    # plt.show(block=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show(block=False)
    plt.savefig(os.path.join(msd_dir, "msd_eigval.png"))


def main():
    aphin_vis.setup_matplotlib(save_plots=False)
    working_dir = os.path.dirname(__file__)
    result_dir = os.path.join(working_dir, "results", "msd_phin_lti_mi")

    # Refactored experiments dictionary using namedtuple
    experiments = {
        "reference": Experiment(
            file_path="reference", linestyle="solid", color="black", marker="o"
        ),
        "lti": Experiment(file_path="lti", linestyle="solid", color="cyan", marker="v"),
        "phin": Experiment(
            file_path="phin", linestyle="solid", color="orange", marker="d"
        ),
        # "mi": Experiment(
        #     file_path="mi/test", linestyle="solid", color="magenta", marker="s"
        # ),
    }

    results = load_experiment_results(result_dir, experiments)
    plot_trajectories(results, experiments)


if __name__ == "__main__":
    main()
