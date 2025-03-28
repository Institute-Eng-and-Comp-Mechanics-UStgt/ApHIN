import os
import logging
import numpy as np
from aphin.utils.data import DiscBrakeDataset
from aphin.utils.configuration import Configuration


def read_abaqus_data_from_txt_files(disc_brake_txt_path):
    """
    Reads and processes Abaqus simulation data from .txt files and saves it into a .npz file for efficient storage.

    This function handles the preprocessing of simulation data obtained from Abaqus, as outlined in the workflow available at:
    https://doi.org/10.18419/darus-4418. The data is originally in .txt file format, which is read and processed by this function.
    The processed data is then saved into a .npz file format for faster access and reduced storage needs. This preprocessing step
    needs to be executed only once unless there are changes to the Abaqus model.

    Parameters:
    -----------
    disc_brake_txt_path : str
        The directory path containing the Abaqus .txt files. This path should lead to the directory where the simulation
        results are stored.

    Returns:
    --------
    None
        The function does not return any values. It processes the input .txt files and logs a summary of the processed data.
    """
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

    disc_brake_data = DiscBrakeDataset.from_txt(
        disc_brake_txt_path,
        save_cache=True,
        cache_path=cache_path,
        idx_mu=idx_mu,
        use_velocities=db_cfg["use_velocities"],
        t_end=db_cfg["t_end"],
    )

    n_sim, n_t, n_n, n_dn, n_u, n_mu = disc_brake_data.shape
    logging.info(
        f"The data was successfully read from the txt files. The data consists of {n_sim} simulations, {n_t} time steps per simulation, {n_n} nodes, {n_dn} DOFs per node, {n_u} inputs and {n_mu} parameters."
    )


if __name__ == "__main__":
    disc_brake_txt_path = "/path/to/txt/folder/"
    disc_brake_txt_path = "/scratch/tmp/jrettberg/Projects/ApHIN_Review/disc_brake_with_hole/abaqus_data_generation/data/job_disc_brake_with_hole_without_force/txt_files/"  # TODO: remove later
    read_abaqus_data_from_txt_files(disc_brake_txt_path)
