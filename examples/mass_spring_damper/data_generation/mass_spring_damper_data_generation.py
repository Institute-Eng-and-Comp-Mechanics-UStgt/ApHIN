# default packages
import numpy as np
import logging
import os
import sys
import yaml
import matplotlib.pyplot as plt

# Add the current script's directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# own packages
import aphin.utils.visualizations as phdl_vis
from parameter_input import ParameterInput
from msd_model import MSD

# set up logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# set up matplotlib
phdl_vis.setup_matplotlib(False)

# %% enter configuration here:
with open(
    os.path.join(os.path.dirname(__file__), "config_data_gen.yml"),
    "r",
) as file:
    msd_config = yaml.safe_load(file)

system_types = ["ph", "ss"]
Q_ids = [True, False]
msd_list = []
for system_type in system_types:
    for Q_id in Q_ids:
        # create ParameterInput instance from config
        parameter_input = ParameterInput.from_config(
            msd_config, seed=msd_config["seed"]
        )

        # create msd example from ParameterInput instance
        msd = MSD.from_parameter_input(parameter_input, system_type=system_type)
        if Q_id:
            if system_type == "ph":
                # transform to Q identity form
                msd.transform_pH_to_Q_identity()
            else:
                continue

        # time values for trajectories
        T_training = msd_config["T_training"]
        time_steps_training = msd_config["time_steps_training"]
        t_training = np.linspace(0, T_training, time_steps_training)

        # convert initial condition to match to appropriate coordinates
        parameter_input = msd.convert_initial_condition(parameter_input)

        # time integration
        msd.solve(
            t=t_training,
            z_init=parameter_input.x0,
            u=parameter_input.u,
            debug=msd_config["debug"],
        )

        # save data and results
        msd.save()
        msd_list.append(msd)

# compare different system formulations - after retransforming they should be the same
msd.compare_msd_systems(msd_list)
