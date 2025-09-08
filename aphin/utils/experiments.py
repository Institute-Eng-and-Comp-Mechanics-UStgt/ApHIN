import os
import itertools
import yaml
import logging
import numpy as np

logger = logging.getLogger(__name__)


def run_various_experiments(
    experiment_main_script,
    parameter_variation_dict,
    basis_config_yml_path,
    result_dir,
    log_dir,
    **kwargs,
):
    """
    Runs multiple experiments by creating several config files.

    Parameters
    ----------
    experiment_main_script : function
        main script that runs the desired example
    parameter_variation_dict: dict
        dict.keys need to match with keyword in basic config file
        dict.values are a list of desired configurations
        e.g. {"scaling_values":["Null",[10,100,0.1]],
              "r": [2,6,10]}
    basis_config_yml_path: (str)
        absolute path to basic config file which shall be manipulated
    result_dir: (str)
        absolute path to where the results

    """

    yaml_paths_list = create_modified_config_files(
        parameter_variation_dict, basis_config_yml_path, result_dir
    )

    run_all_yaml_files(
        experiment_main_script,
        yaml_paths_list,
        # result_dir,
        log_dir,
        **kwargs,
    )


def create_modified_config_files(
    parameter_variation_dict, basis_config_yml_path, result_dir
):
    """
    Generate modified YAML configuration files based on variations in parameter values.

    This function takes a base configuration file and creates new configuration files for each combination of parameter values provided. Each generated configuration file is named uniquely based on the experiment name and parameter values. If a parameter value is a float, it is formatted to avoid scientific notation and unnecessary decimal points.

    Parameters:
    -----------
    parameter_variation_dict : dict
        Dictionary where keys are configuration parameter names and values are lists of possible values for those parameters. The function will generate configuration files for every combination of these parameter values.

    basis_config_yml_path : str
        File path to the base YAML configuration file used as the template for modifications.

    result_dir : str
        Directory where the modified configuration files will be saved.

    Returns:
    --------
    yaml_paths_list : list of str
        List of file paths to the generated YAML configuration files.
    """
    # %% generate config files
    yaml_paths_list = []
    for experiment in dict_product(parameter_variation_dict):
        # %% create save name
        # basic config .yml file
        # basis_config_yml_path = os.path.join(working_dir, "../config.yml")
        cfg = yaml.safe_load(open(basis_config_yml_path))
        experiment_name_as_prefix = cfg["experiment"]

        save_name = f"{experiment_name_as_prefix}_"
        for key, value in experiment.items():
            if isinstance(value, list):
                value_str = ""
                for string in value:
                    value_str += f"{string}_"
                value_str = value_str.strip("_")
                save_name += f"{key}_{value_str}_"
            else:
                save_name += f"{key}_{value}_"
        save_name = save_name.replace(".", "_")
        save_name = save_name.strip("_")
        # rename experiment to make it unique
        experiment["experiment"] = save_name
        # %% create config files for different experiment runs
        keyword_found_list = []
        adapted_config_file_path = os.path.join(result_dir, f"{save_name}.yml")
        with open(basis_config_yml_path, "r") as basic_config_file:
            with open(
                adapted_config_file_path, "w"
            ) as adapted_config_file:  # basis config.yml
                # loop over lines
                for line_num, line in enumerate(basic_config_file):
                    # loop over keys
                    for key, value in experiment.items():
                        # check for keyword in config file
                        if line.startswith(f"{key}:"):
                            keyword_found_list.append(key)
                            split_comment = line.split(
                                "#"
                            )  # check for comments and add them
                            if isinstance(value, float):
                                # omit scientific notation
                                # strip '.' at the end if value has no decimals
                                value = np.format_float_positional(value).strip(".")
                            line = f"{key}: {value} \n"
                            if len(split_comment) > 1:
                                line += f"# {split_comment[-1]}"  # add comment
                    adapted_config_file.write(line)
                if not all(
                    ele in keyword_found_list for ele in parameter_variation_dict.keys()
                ):
                    raise ValueError(f"Not all keys have been found in config.yml.")
        yaml_paths_list.append(adapted_config_file_path)
    return yaml_paths_list


def find_all_yaml_files(working_dir):
    """
    Locate all YAML files (.yml) within a specified directory and its subdirectories.

    This function searches through the given working directory and all its subfolders to find files with a `.yml` extension.

    Parameters:
    -----------
    working_dir : str
        The root directory where the search for YAML files begins.

    Returns:
    --------
    yaml_files : list of str
        A list of file paths to all YAML files found within the working directory and its subfolders.
    """
    yaml_files = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(working_dir)
        for f in filenames
        if f.endswith(".yml")
    ]
    return yaml_files


def run_all_yaml_files(
    experiment_main_script,
    yaml_paths_list,
    log_dir,
    **kwargs,
):
    """
    Execute a main experiment script for each YAML configuration file and log the results.

    This function iterates over a list of YAML configuration file paths, executes the provided
    `experiment_main_script` for each configuration, and logs the results to a separate log file
    for each experiment.

    Parameters:
    -----------
    experiment_main_script : callable
        A function that executes the main experiment. It should accept a `config_path_to_file`
        keyword argument, which is the path to the YAML configuration file. This function runs
        the experiment based on the provided configuration.

    yaml_paths_list : list of str
        A list of file paths to YAML configuration files. Each file specifies a different set
        of parameters for the experiment.

    log_dir : str
        Directory where the log files will be saved. Each experiment will have a corresponding
        log file named after the experiment.

    Returns:
    --------
    None
        This function does not return any value.
    """
    # %% loop over config files
    for i, yaml_file_path in enumerate(yaml_paths_list):

        # get experiment name
        cfg = yaml.safe_load(open(yaml_file_path))
        experiment_name = cfg["experiment"]
        # remove old file handler
        if not i == 0:
            logger.removeHandler(file_handler)
        # setup logger
        logger_file_name = os.path.join(log_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(logger_file_name)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        try:
            experiment_main_script(config_path_to_file=yaml_file_path, **kwargs)
        except Exception as e:
            logger.error(f"Run ended with error {e}")


def dict_product(d):
    """
    Generate all possible combinations of parameter values from a dictionary.

    This function yields all combinations of the values in the dictionary `d`, where each key
    in the dictionary corresponds to a parameter with multiple possible values. The combinations
    are generated by creating a Cartesian product of the values for all keys.

    Parameters:
    -----------
    d : dict
        A dictionary where each key is a parameter and the corresponding value is a list of possible
        values for that parameter. The function will generate all combinations of these values.

    Yields:
    -------
    dict
        A dictionary representing a single combination of parameter values. Each dictionary has the same
        keys as the input dictionary `d`, with values corresponding to one combination of the possible
        values.

    Examples:
    ---------
    >>> param_dict = {'param1': [1, 2], 'param2': ['a', 'b']}
    >>> list(dict_product(param_dict))
    [{'param1': 1, 'param2': 'a'}, {'param1': 1, 'param2': 'b'}, {'param1': 2, 'param2': 'a'}, {'param1': 2, 'param2': 'b'}]
    """
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))
