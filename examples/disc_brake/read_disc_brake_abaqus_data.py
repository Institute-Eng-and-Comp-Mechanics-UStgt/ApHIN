import os
import logging
import numpy as np
from aphin.utils.data import DiscBrakeDataset
from aphin.utils.configuration import Configuration


def main():
    """
    Following the workflow to obtain simulation data from Abaqus, see https://doi.org/10.18419/darus-4418
    We obtain the data in form of .txt files
    This function reads and processes the .txt files.
    The data is saved into a more convenient (faster reading, less storage) .npz file.
    Hence, this preprocessing function needs to be executed only once or if changes in the Abaqus model were conducted.

    Please define the path to the folder with the .txt files below.
    """
    # -----------------------------------------
    # USER INPUT REQUIRED
    # -----------------------------------------
    disc_brake_txt_path = "/path/to/txt/folder/"

    # setup experiment based on config file
    working_dir = os.path.dirname(__file__)
    configuration = Configuration(working_dir, config_info=None)
    db_cfg = configuration.cfg_dict
    data_dir, _, _, _ = configuration.directories

    # save/load data path
    sim_name = db_cfg["sim_name"]
    cache_path = os.path.join(data_dir, f"{sim_name}.npz")  # path to .npz file

    # create data from .txt files
    # idx of parameters in parameter file
    idx_mu = np.arange(db_cfg["n_mu"])

    disc_brake_txt_path = os.path.join(disc_brake_txt_path, sim_name)
    disc_brake_data = DiscBrakeDataset.from_txt(
        disc_brake_txt_path,
        save_cache=True,
        cache_path=cache_path,
        idx_mu=idx_mu,
        use_velocities=db_cfg["use_velocities"],
    )

    n_sim, n_t, n_n, n_dn, n_u, n_mu = disc_brake_data.shape
    logging.info(
        f"The data was successfully read from the txt files. The data consists of {n_sim} simulations, {n_t} time steps per simulation, {n_n} nodes, {n_dn} DOFs per node, {n_u} inputs and {n_mu} parameters."
    )


if __name__ == "__main__":
    main()
