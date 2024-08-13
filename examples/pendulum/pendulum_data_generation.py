import numpy as np
import logging
import os
from scipy.stats import qmc
from scipy.integrate import solve_ivp

from phdl.utils.configuration import Configuration

# set up logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# setup experiment based on config file
working_dir = os.path.dirname(__file__)
configuration = Configuration(working_dir, config_info=None)
pd_cfg = configuration.cfg_dict
data_dir, log_dir, weight_dir, result_dir = configuration.directories

# %% Script functions


# ODE in generalized coordinates phi
def fun_ode(t, X):
    phi = X[0]
    phi_dot = X[1]
    return np.array([phi_dot, -g / r0 * np.sin(phi)])


# %% Read simulated disc brake data
logging.info(
    "################################   1. Data       ################################"
)
t = np.linspace(0, 10, pd_cfg["n_t"])[:, np.newaxis]

# Physical parameters for the pendulum
phi_0_range = pd_cfg["phi_0"]  # initial angle
phi_dot0_range = pd_cfg["phi_dot0"]  # initial angular velocity

# sample initial conditions using Halton sequence
sampler = qmc.Halton(2, seed=pd_cfg["seed"])
initial_conditions = sampler.random(pd_cfg["n_sim"])
x_init_ode = qmc.scale(
    sampler.random(pd_cfg["n_sim"]),
    [phi_0_range[0], phi_dot0_range[0]],
    [phi_0_range[1], phi_dot0_range[1]],
)


# we use the same parameters for all simulations (no parametric usecase)
r0 = pd_cfg["r0"]  # rod length
m = pd_cfg["m"]  # mass
g = pd_cfg["g"]  # gravitational acceleration

n_t = t.shape[0]
n_sim = pd_cfg["n_sim"]
X = np.empty((pd_cfg["n_sim"], pd_cfg["n_t"], pd_cfg["n_n"], pd_cfg["n_dn"]))
X_dt = np.empty((pd_cfg["n_sim"], pd_cfg["n_t"], pd_cfg["n_n"], pd_cfg["n_dn"]))
for i, x_init_ode in enumerate(initial_conditions):
    logging.info(f"Simulation {i+1}/{pd_cfg['n_sim']}")
    sol_ode = solve_ivp(
        fun=fun_ode,
        t_span=(t.min(), t.max()),
        max_step=t.max() / 10,
        t_eval=t.ravel(),
        y0=x_init_ode,
        dense_output=True,
    )
    phi = sol_ode.y[0, :]
    phi_dot = sol_ode.y[1, :]

    # Reconstruct Cartesian coordinates based on the ODE's solution
    x_ode = r0 * np.sin(phi)
    y_ode = -r0 * np.cos(phi)
    vx_ode = r0 * phi_dot * np.cos(phi)
    vy_ode = r0 * phi_dot * np.sin(phi)
    X[i] = np.expand_dims((np.vstack([x_ode, y_ode, vx_ode, vy_ode]).T), axis=1)
    X_dt[i] = np.gradient(X[i], t.ravel(), axis=0)

# %% save data
np.savez(
    os.path.join(data_dir, "pendulum.npz"),
    X=X,
    X_dt=X_dt,
    t=t,
)
