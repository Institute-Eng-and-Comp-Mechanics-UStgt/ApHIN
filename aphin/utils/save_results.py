import os
import numpy as np
import shutil
import pandas as pd


def save_results(weight_dir, path_to_config, config_dict, result_dir, load_network):
    """
    calls the functions to save weights, config and writes to overview .csv

    Parameters:
    -----------
    weight_dir : str
        Directory where the neural network weights are saved.
    path_to_config : str
        The path to the configuration file (`config.yml`). This file should be a YAML file with
        the experiment's configuration.
    config_dict : dict
        Configuration dictionary, typically obtained from the Configuration class.
    result_dir : str
        Destination directory where the results are stored. The overview CSV is written to the parent directory.
    """
    save_weights(weight_dir, result_dir, load_network)
    save_config(path_to_config, result_dir, load_network)
    write_to_experiment_overview(config_dict, result_dir, load_network)

def save_evaluation_times(data_id, result_dir):
    """
    Save the evaluation times to a CSV file.

    This function saves the evaluation times of the model to a CSV file in the specified `result_dir`.

    Parameters:
    -----------
    data_id : Identified dataset of class aphin.utils.data.Dataset
        DataIdentification object that contains the evaluation times.
    result_dir : str
        Destination directory where the results are stored.
    """

    for key, data_ in dict(train=data_id.TRAIN, test=data_id.TEST).items():
        file_name = f"evaluation_times_{key}.csv"
        file_dir = os.path.join(result_dir, file_name)
        # print(range(1, data_.n_sim + 1))
        header = ",".join([str(i) for i in list(range(1, data_.n_sim + 1))])
        header = "mean," + header
        solving_times = np.insert(data_.solving_times['per_run'], 0, data_.solving_times['mean'])[np.newaxis]

        np.savetxt(
            file_dir,
            solving_times,
            delimiter=",",
            fmt="%s",
            comments="",
            header=header,
        )

def save_training_times(tran_hist, result_dir):
    """
    Save the training times to a CSV file.

    Parameters:
    -----------
    tran_hist : History object from a Keras model
        History object that contains the training times.
    result_dir : str
        Destination directory where the results are stored.
    """
    file_name = "training_time.csv"
    file_dir = os.path.join(result_dir, file_name)
    header = "epochs,time,time_per_epoch"
    values = np.array([str(tran_hist.params['epochs']),
                       str(tran_hist.history['time']),
                       str(tran_hist.history['time_per_epoch'])])[np.newaxis]
    np.savetxt(
        file_dir,
        values,
        delimiter=",",
        fmt="%s",
        comments="",
        header=header,
    )


def save_weights(weight_dir, result_dir, load_network):
    """
    Copy neural network weights from the weight directory to the result directory.

    This function copies the weights file from `weight_dir` to `result_dir`, ensuring that all relevant data
    is consolidated in one location. If `load_network` is True, no action is taken.

    Parameters:
    -----------
    weight_dir : str
        Directory where the neural network weights are saved.
    result_dir : str
        Destination directory where the results are stored.
    load_network : bool
        Boolean indicating if the network is loaded in this experiment run. If True, weights are not copied.
    """
    if load_network:
        pass
    else:
        # copy weights
        shutil.copy2(
            os.path.join(weight_dir, ".weights.h5"),
            os.path.join(result_dir, ".weights.h5"),
        )


def save_config(path_to_config, result_dir, load_network):
    """
    Copies the configuration file to the results directory to consolidate all relevant data.

    This function ensures that the configuration file (`config.yml`) is copied to the specified
    `result_dir` for convenience. This helps in keeping all experiment data and configuration
    in a single folder. If `load_network` is `True`, the function does not perform any action.

    Parameters:
    -----------
    path_to_config : str
        The path to the configuration file (`config.yml`). This file should be a YAML file with
        the experiment's configuration.

    result_dir : str
        The destination directory where the results are stored. The configuration file will be
        copied to this directory.

    load_network : bool
        A flag indicating whether a network is loaded in this experiment run. If `True`, no action
        is taken. If `False`, the configuration file is copied.

    Returns:
    --------
    None
        The function does not return any value. It performs a file copy operation.
    """
    if load_network:
        pass
    else:
        # copy config.yml
        if not (os.path.isfile(path_to_config) and path_to_config.endswith(".yml")):
            raise ValueError(f"Path {path_to_config} does not lead to config file.")
        shutil.copy2(
            path_to_config,
            os.path.join(result_dir, "config.yml"),
        )


def write_to_experiment_overview(config_dict, result_dir, load_network):
    """
    Append experiment configuration details to an overview CSV file.

    This function updates a CSV file named `experiments_overview.csv` with the configuration details of
    the current experiment. If `load_network` is True, no new entry is added. If False, the configuration
    is added to the overview.

    Parameters:
    -----------
    config_dict : dict
        Configuration dictionary, typically obtained from the Configuration class.
    result_dir : str
        Destination directory where the results are stored. The overview CSV is written to the parent directory.
    load_network : bool
        Boolean indicating if the network is loaded in this experiment run. If True, no new entry is added to the CSV.
    """

    if load_network:
        # only create new entry if network is not loaded
        pass
    else:
        # name of csv file
        csv_filename = "experiments_overview.csv"

        # check if overview exists
        # result_dir uses experiment name -> use parent directory level
        path_to_csv = os.path.join(result_dir, "..", csv_filename)

        if os.path.isfile(path_to_csv):
            # read existing csv
            df_from_csv = pd.read_csv(path_to_csv)

        else:
            # create new overview
            # empty dataframe
            df_from_csv = pd.DataFrame({})

        # write config data to dataframe
        # to avoid 'differnt length' error use "orient='index'" and transpose afterwards
        df_config = pd.DataFrame.from_dict(config_dict, orient="index").transpose()

        df_concatenated = pd.concat(
            [df_from_csv, df_config], ignore_index=True, sort=False
        )

        # write to csv
        df_concatenated.to_csv(path_to_csv, index=False)
