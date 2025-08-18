import os
import matplotlib.pyplot as plt
import numpy as np
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis

#%% This script assumes that the results of the pendulum example have been run
# and stored in the folder `results/pendulum/`.

aphin_vis.setup_matplotlib(save_plots=False)

# setup experiment based on config file
working_dir = os.path.dirname(__file__)
configuration = Configuration(working_dir)
pd_cfg = configuration.cfg_dict
data_dir, log_dir, weight_dir, result_dir = configuration.directories

# load results
experiments = dict(reference=["phin/reference.csv", "solid", "black"],
              phin=["phin/phin.csv", "dashdot", "cyan"],
              aphin_linear=["aphin_linear/aphin_linear.csv", "dotted", "purple"],
              aphin_nonlinear=["aphin_nonlinear/aphin_nonlinear.csv", "dashed", "magenta"])

n_trajectories = 6
n_states = 2
results = {}

for key, experiment in experiments.items():
    file_dir = os.path.join(result_dir, "..", experiment[0])
    if not os.path.exists(file_dir):
        raise FileNotFoundError(f"Results file for {experiment} not found at {result_dir}")
    # load csv file
    data = np.genfromtxt(file_dir, delimiter=',', skip_header=1)
    t = data[:,0]
    x = data[:, 1:n_states*n_trajectories+1].reshape(1000, n_trajectories, 2).transpose([1,0,2])
    dxdt = data[:, n_states*n_trajectories+1:(2*n_states*n_trajectories+1)].reshape(1000, n_trajectories, 2).transpose([1,0,2])
    dxddt = data[:, (2*n_states*n_trajectories+1):].reshape(1000, n_trajectories, 2).transpose([1,0,2])
    results[key] = dict(t=t, x=x, dxdt=dxdt, dxddt=dxddt)


# plot trajectories
i_traj = 0
# activate latex
fig, axs = plt.subplots(3,2)

for (i, result), (_, experiment) in zip(results.items(), experiments.items()):
    for i_state, state_name in enumerate(["x", "y"]):
        axs[0, i_state].plot(result["t"], result["x"][i_traj,:,i_state], linestyle=experiment[1], color=experiment[2])
        axs[1, i_state].plot(result["t"], result["dxdt"][i_traj,:,i_state], linestyle=experiment[1], color=experiment[2])
        axs[2, i_state].plot(result["t"], result["dxddt"][i_traj,:,i_state], linestyle=experiment[1], color=experiment[2])
        # labels
        axs[0, i_state].set_ylabel(r"${\rho}_"+state_name+"$")
        axs[1, i_state].set_ylabel(r"$\dot{\rho}_"+state_name+"$")
        axs[2, i_state].set_xlabel("time in s")
        axs[2, i_state].set_ylabel(r"$\ddot{\rho}_"+state_name+"$")
# outside legend north of plots
fig.legend(list(experiments.keys()), loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2)
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Increase the top margin to make space for the legend
plt.show()