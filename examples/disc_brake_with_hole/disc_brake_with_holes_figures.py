import os
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib.colors import to_rgba

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

def plot_trajectories(results):
    """Plot state, velocity, and acceleration trajectories."""
    colors = ["magenta", "cyan", "purple"]
    # state trajectory plot
    data = results["r12"]
    fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharex=True)
    for i, (quantity, c) in enumerate(zip(["temp", "disp", "vel"], colors)):
        for j in range(N_STATES):
            axs[i,0].plot(data["t"], data[f"ref_{quantity}"][:,j], linestyle="dashed", color="black", label="Reference", zorder=5)
            axs[i,0].plot(data["t"], data[f"{quantity}"][:,j], linestyle="solid", color=c, label="Experiment r12")
        # for j in range(N_TRAJECTORIES):
        axs[i,1].semilogy(data["t"], data[f"rms_{quantity}"].mean(axis=0),
                    color=c, zorder=10)
        axs[i,1].semilogy(data["t"], data[f"rms_{quantity}"].transpose(),
                    color="gray")
        axs[i, 1].set_ylabel(rf"rel RMS error")

    axs[0,0].set_title(r"$\bm{\vartheta}$")
    axs[0,0].set_ylabel(r"temp. in °C")
    axs[1,0].set_title(r"$\bm{q}_z$")
    axs[1,0].set_ylabel(r"disp. in m")
    axs[2,0].set_title(r"$\dot{\bm{q}}_z$")
    axs[2,0].set_ylabel(r"vel in m/s")
    axs[2,0].set_xlabel("time in s")
    axs[0,1].set_title(r"$\bm{\vartheta}$")
    axs[1,1].set_title(r"$\bm{q}$")
    axs[2,1].set_title(r"$\dot{\bm{q}}$")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    axs[2,0].legend(["reference", "identified"], loc='lower center', bbox_to_anchor=(0.5, -0.65), ncol=1)
    axs[2,1].legend(["mean", "all test scenarios"], loc='lower center', bbox_to_anchor=(0.5, -0.65), ncol=1)


    # boxplot
    error_vel = [r["rms_vel"].mean(axis=1) for r in results.values()]
    error_disp = [r["rms_disp"].mean(axis=1) for r in results.values()]
    error_temp = [r["rms_temp"].mean(axis=1) for r in results.values()]
    r_levels = np.array([2, 4, 8, 12, 16, 24])

    fig = plt.figure(figsize=(2.5, 6.5), dpi=200)  # tall & narrow like the sample
    ax = fig.add_subplot(111)

    series_names = [r"$\vartheta$", "q",  r"$\dot{q}$"]
    series_data = [error_temp, error_disp, error_vel]

    # group/box positions
    x_centers = np.arange(1, len(r_levels) + 1)  # 1..6
    group_width = 0.9
    box_w = group_width / 3.5
    offsets = np.array([-box_w * 1.1, 0.0, box_w * 1.1])

    # Draw subtle vertical guides per group (like your plot)
    for xc in x_centers:
        ax.axvline(xc+0.5, color="#dddddd", lw=0.6, zorder=0)

    # Plot each series
    for k, (name, data) in enumerate(zip(series_names, series_data)):
        xs = x_centers + offsets[k]

        ax.boxplot(
            data,
            positions=xs,
            widths=box_w,
            patch_artist=True,
            showfliers=True,  # keep small outliers
            boxprops=dict(linewidth=0.8, facecolor=to_rgba(colors[k], alpha=0.15), edgecolor=colors[k]),
            medianprops=dict(color=colors[k], linewidth=1.0),
            whiskerprops=dict(color=colors[k], linewidth=0.8),
            capprops=dict(color=colors[k], linewidth=0.8),
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor=colors[k],
                            markeredgecolor="none", alpha=0.8),
            label= name
        )


    # Axes labels/ticks to match style
    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(r) for r in r_levels])
    ax.set_xlabel(r"reduced order $r$", fontsize=12)
    ax.set_ylabel("error", fontsize=12)

    # Legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 0.93),
              frameon=True, framealpha=0.95, edgecolor="#aaaaaa")

    fig.tight_layout()
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
    plot_trajectories(results)

if __name__ == "__main__":
    main()