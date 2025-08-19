import os
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis
from matplotlib.lines import Line2D

# Constants
N_STATES = 5
N_T = 2000
N_TRAJECTORIES = 10
TRAJECTORY_INDEX = 3

# Define a named tuple for experiment properties
Experiment = namedtuple("Experiment", ["file_path"])

def load_experiment_results(result_dir, experiments):
    """Load results for all experiments."""
    results = {}
    for key, experiment in experiments.items():
        file_dir = os.path.join(result_dir, "..", experiment.file_path)
        if not os.path.exists(file_dir):
            raise FileNotFoundError(f"Results file for experiment '{key}' not found at {file_dir}")
        # rms error
        data = np.genfromtxt(os.path.join(file_dir, "rms_error_state_disp.csv"), delimiter=',', skip_header=1).transpose()
        t = data[0]
        rms_disp = data[1:N_TRAJECTORIES + 1]
        rms_temp = np.genfromtxt(os.path.join(file_dir, "rms_error_state_temp.csv"), delimiter=',', skip_header=1).transpose()[1:N_TRAJECTORIES + 1]
        rms_vel = np.genfromtxt(os.path.join(file_dir, "rms_error_state_vel.csv"), delimiter=',', skip_header=1).transpose()[1:N_TRAJECTORIES + 1]

        # state trajectories
        state_traj_file = os.path.join(file_dir, "db_custom_nodes.csv")
        if os.path.exists(state_traj_file):
            data = np.genfromtxt(os.path.join(file_dir, "db_custom_nodes.csv"), delimiter=',', skip_header=1)
            ref_temp = data[:, 1:1+N_STATES]
            ref_displacement = data[:, 1+N_STATES:1+2*N_STATES]
            ref_vel = data[:, 1+2*N_STATES:1+3*N_STATES]
            temp = data[:, 1+3*N_STATES:1+4*N_STATES]
            displacement = data[:, 1+4*N_STATES:1+5*N_STATES]
            velocity = data[:, 1+5*N_STATES:1+6*N_STATES]
        else:
            ref_temp, ref_displacement, ref_vel = None, None, None
            temp, displacement, velocity = None, None, None

        results[key] = dict(t=t, rms_disp=rms_disp, rms_temp=rms_temp, rms_vel=rms_vel,
                            ref_temp=ref_temp, ref_disp=ref_displacement, ref_vel=ref_vel,
                           temp=temp, disp=displacement, vel=velocity)
    return results

def plot_trajectories(results, experiments):
    """Plot state, velocity, and acceleration trajectories."""
    # %% state trajectory plot
    data = results["r12"]
    fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharex=True)
    for i, (quantity, c) in enumerate(zip(["vel", "disp", "temp"], ["purple", "cyan", "magenta"])):
        for j in range(N_STATES):
            axs[i,0].plot(data["t"], data[f"ref_{quantity}"][:,j], linestyle="solid", color="black", label="Reference")
            axs[i,0].plot(data["t"], data[f"{quantity}"][:,j], linestyle="dashed", color=c, label="Experiment r12")
        # for j in range(N_TRAJECTORIES):
        axs[i,1].semilogy(data["t"], data[f"rms_{quantity}"].mean(axis=0),
                    color=c, zorder=10)
        axs[i,1].semilogy(data["t"], data[f"rms_{quantity}"].transpose(),
                    color="gray")
        axs[i, 1].set_ylabel(rf"rel RMS error")

    axs[0,0].set_title(r"$\dot{\bm{q}}_z$")
    axs[0,0].set_ylabel(r"vel in m/s")
    axs[1,0].set_title(r"$\bm{q}_z$")
    axs[1,0].set_ylabel(r"disp. in m")
    axs[2,0].set_title(r"$\bm{\theta}$")
    axs[2,0].set_xlabel("time in s")
    axs[2,0].set_ylabel(r"temp. in °C")
    axs[0,1].set_title(r"$\bm{q}$")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    axs[2,0].legend(["reference", "identified"], loc='lower center', bbox_to_anchor=(0.5, -0.65), ncol=1)
    axs[2,1].legend(["all test scenarios", "mean"], loc='lower center', bbox_to_anchor=(0.5, -0.65), ncol=1)
    plt.show()

        # ax[0].plot(results["reference"]["t"], results["reference"][f"rms_{quantity}"].mean(axis=0), linestyle="solid", color="black", label="Reference")
        # experiment = experiments[key]
        # # Eigenvalue plot
        # eigenvalues = result["eigenvalues"]
        # axs[0].scatter(eigenvalues[TRAJECTORY_INDEX, :, 0], eigenvalues[TRAJECTORY_INDEX, :, 1], color=experiment.color, marker=experiment.marker, label=key)
        # # state trajectory plot
        # axs[1].plot(result["t"], result["x"][TRAJECTORY_INDEX, :, :], linestyle=experiment.linestyle, color=experiment.color, label=key)
        # # error plot
        # if result["t_error"] is not None:
        #     for i in range(N_TRAJECTORIES):
        #         axs[2].plot(result["t_error"], result["state_error"][i, :], linestyle=experiment.linestyle,  alpha=0.03, color=experiment.color, label='_nolegend_')
        #     axs[2].plot(result["t_error"], result["state_error"].mean(axis=0), linestyle=experiment.linestyle, color=experiment.color, label='_nolegend_', zorder=10)

    axs[0].set_xlabel("$Re(\lambda)$")
    axs[0].set_ylabel("$Im(\lambda)$")
    axs[1].set_xlabel("time in s")
    axs[1].set_ylabel(r"$\bm{q}$")
    axs[2].set_xlabel("time in s")
    axs[2].set_ylabel("rel RMS error")

    # Create custom legend
    custom_lines = [
        Line2D([0], [0], color=experiments["reference"].color, linestyle=experiments["reference"].linestyle, label="Reference"),
        Line2D([0], [0], color=experiments["lti"].color, linestyle=experiments["lti"].linestyle, label="LTI"),
        Line2D([0], [0], color=experiments["phin"].color, linestyle=experiments["phin"].linestyle, label="PHIN"),
        Line2D([0], [0], color=experiments["mi"].color, linestyle=experiments["mi"].linestyle, label="MI"),
    ]
    fig.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def main():
    aphin_vis.setup_matplotlib(save_plots=False)
    working_dir = os.path.dirname(__file__)
    configuration = Configuration(working_dir)
    _, _, _, result_dir = configuration.directories
    reduced_orders = [2,4,8,12,16,24]
    # Refactored experiments dictionary using namedtuple
    experiments = {
        f"r{reduced_order}": Experiment(file_path=f"db_large_heat_larger_data_epoch6000_rsweep_multExp_r{reduced_order}") for reduced_order in reduced_orders
    }

    results = load_experiment_results(result_dir, experiments)
    plot_trajectories(results, experiments)

if __name__ == "__main__":
    main()