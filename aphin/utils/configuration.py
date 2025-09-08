import yaml
import os
import logging
import datetime
from aphin.utils.save_results import save_config


class Configuration:
    """
    Handles the setup and management of configuration for an experiment.

    This class manages the reading of configuration files, creation of necessary directories,
    and saving of the configuration to ensure all necessary data is stored and organized. It
    supports both identifying and loading results based on the provided configuration information.
    """

    def __init__(
        self, working_dir, config_info=None, overwrite_results=False, various_exp=False
    ) -> None:
        """
        Initializes the Configuration instance.

        Sets up the working directory, reads the configuration file, creates necessary directories,
        and saves the configuration file. The behavior of this method depends on the provided
        `config_info` parameter, which determines whether to use default configurations,
        specific configuration files, or result directories.

        Parameters:
        -----------
        working_dir : str
            Absolute path to the working directory where all needed folders are set up.
        config_info : str or None
            Specifies the source of the configuration file or directory:
            - None: Uses the default `config.yml` in the working directory for identification.
                    If `load_network` is True in the config, it checks for the existence of
                    weights in the default path.
            - config_filename.yml: An absolute path to a custom configuration file ending with `.yml` for identification.
                                It uses the specified config file, and if `load_network` is True,
                                checks for the weights in the file path.
            - /folder/name/: An absolute path to a directory containing `config.yml` and `.weights.h5`.
                            This is used for loading results.
            - result_folder_name: A folder name under the working directory that contains
                                `config.yml` and `.weights.h5`. This is also used for loading results.
        overwrite_results : bool, optional
            Flag indicating whether to overwrite existing results if they are found in the result directory.
            Defaults to `False`.

        Returns:
        --------
        None
            Initializes the instance and sets up necessary configurations and directories.
        """
        self.working_dir = working_dir
        self.datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # read config and get weight directory
        self.read_config(config_info, various_exp=various_exp)
        # check for mandatory config entries
        self.check_config_dict()
        # create directories
        self.create_directories(overwrite_results=overwrite_results)
        # save configuration (if not load_network)
        self.save_config()

    @property
    def directories(self):
        """
        Returns the paths to the directories used in the experiment setup.

        This property provides access to the directory paths for data, logs, weights, and results.

        Returns:
        --------
        tuple of str
            A tuple containing the paths to the following directories:
            - `data_dir`: Directory for data storage.
            - `log_dir`: Directory for log files.
            - `weight_dir`: Directory for storing model weights.
            - `result_dir`: Directory for storing results.
        """
        return self.data_dir, self.log_dir, self.weight_dir, self.result_dir

    def read_config(self, config_info, various_exp=False):
        """
        Reads and loads configuration settings from a specified source, determining whether to proceed
        with model identification or to load existing results. The method sets up paths to configuration
        files, weight directories, and checks the existence of necessary files.

        Parameters:
        -----------
        config_info : str or None
            Specifies the source of the configuration file or directory:
            - None: Uses the default `config.yml` in the working directory for identification.
                    If `load_network` is True in the config, it checks for the existence of
                    weights in the default path.
            - config_filename.yml: An absolute path to a custom configuration file ending with `.yml` for identification.
                                It uses the specified config file, and if `load_network` is True,
                                checks for the weights in the file path.
            - /folder/name/: An absolute path to a directory containing `config.yml` and `.weights.h5`.
                            This is used for loading results.
            - result_folder_name: A folder name under the working directory that contains
                                `config.yml` and `.weights.h5`. This is also used for loading results.

        Returns:
        --------
        None
            This method sets up internal paths and configuration dictionary (self.cfg_dict) based on the provided input.
        """
        assert config_info is None or isinstance(config_info, str)

        if config_info is None:
            # %% default config file for identification
            self.path_to_config = os.path.join(self.working_dir, "config.yml")
            logging.info(f"Using default configuration from {self.path_to_config}.")
            self.cfg_dict = yaml.safe_load(open(self.path_to_config))

            # default weights folder
            self.weight_dir = os.path.join(
                self.working_dir, "weights", self.cfg_dict["experiment"]
            )

            # check if weights exist if network is loaded
            if self.cfg_dict["load_network"] and not various_exp:
                assert os.path.isfile(os.path.join(self.weight_dir, ".weights.h5"))

        elif config_info.endswith(".yml"):
            # %% manually given config.yml file for identification
            assert os.path.isfile(config_info)
            self.path_to_config = config_info
            logging.info(f"Using specified configuration from {self.path_to_config}.")
            self.cfg_dict = yaml.safe_load(open(self.path_to_config))

            # default weights folder
            self.weight_dir = os.path.join(
                self.working_dir, "weights", self.cfg_dict["experiment"]
            )

            # check if weights exist if network is loaded
            if self.cfg_dict["load_network"] and not various_exp:
                # check if .weights.h5 file is in weight_dir or subfolders
                assert os.path.isfile(
                    os.path.join(self.weight_dir, ".weights.h5")
                ) or any(
                    os.path.isfile(os.path.join(self.weight_dir, i, ".weights.h5"))
                    for i in os.listdir(self.weight_dir)
                )

        elif os.path.isdir(config_info):
            # %% path to result folder in which results are stored
            self.path_to_config = os.path.join(config_info, "config.yml")
            path_to_weights = os.path.join(config_info, ".weights.h5")
            assert os.path.isfile(self.path_to_config)
            assert os.path.isfile(path_to_weights)
            self.weight_dir = config_info
            self.cfg_dict = yaml.safe_load(open(self.path_to_config))

        else:
            # %% result folder name somewhere under working_dir
            self.path_to_config = None
            for root, dirs, _ in os.walk(self.working_dir):
                if config_info in dirs:
                    path_to_config_dir = os.path.join(root, config_info)
                    self.path_to_config = os.path.join(path_to_config_dir, "config.yml")
                    self.cfg_dict = yaml.safe_load(open(self.path_to_config))
                    path_to_weights = os.path.join(path_to_config_dir, ".weights.h5")
                    assert os.path.isfile(
                        os.path.join(path_to_config_dir, ".weights.h5")
                    ) or any(
                        os.path.isfile(
                            os.path.join(path_to_config_dir, folder, ".weights.h5")
                        )
                        for folder in os.listdir(path_to_config_dir)
                    )  # check if weights exist in folder or subfolders
                    self.weight_dir = path_to_config_dir
                    # set load_network to true, since it is loaded
                    self.cfg_dict["load_network"] = True
                    self.load_results = True
                    break  # stop after result folder has been found

            if self.path_to_config is None:
                raise ValueError(f"The result folder {config_info} was not found.")

    def create_directories(self, overwrite_results=False):
        """
        Creates necessary directories for storing data, logs, weights, and results based on the current configuration.
        If the directories already exist, they are preserved unless `overwrite_results` is explicitly set to True.

        Parameters:
        -----------
        overwrite_results : bool, optional
            If set to True, existing result directories and files (e.g., `.weights.h5`) are overwritten.
            Default is False, meaning that the method will abort if existing results are found.

        Returns:
        --------
        None
            This method does not return any value. It sets up the directory structure required for the experiment.
        """

        experiment = self.cfg_dict["experiment"]

        self.data_dir = os.path.join(self.working_dir, "data")
        self.log_dir = os.path.join(
            self.working_dir, "logs", experiment, self.datetime_str
        )
        self.result_dir = os.path.join(self.working_dir, "results", experiment)

        # create paths if they do not exist
        for path in [self.data_dir, self.log_dir, self.weight_dir, self.result_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        # default: do not overwrite results
        if (
            os.path.isfile(os.path.join(self.result_dir, ".weights.h5"))
            and self.cfg_dict["load_network"] is False
            and not (overwrite_results)
        ):
            logging.error(
                f"Results in {self.result_dir} already exist. Run is aborted."
            )

    def check_config_dict(self):
        """
        Validates the configuration dictionary by ensuring all mandatory keys are present.

        This method checks that the configuration dictionary (`self.cfg_dict`) contains all required entries
        necessary for the proper functioning of the experiment setup. If any mandatory key is missing,
        an exception is raised, prompting the user to update the configuration file.

        Parameters:
        -----------
        None

        Raises:
        -------
        ValueError
            If any mandatory key is missing from `self.cfg_dict`, a `ValueError` is raised,
            specifying the missing key.

        Notes:
        ------
        - The method checks for the presence of the following mandatory keys in `self.cfg_dict`:
            - `experiment`: Specifies the name of the experiment.
            - `load_network`: Indicates whether to load a pre-existing network.
        - This validation ensures that critical configuration settings are not overlooked, helping prevent runtime errors due to missing configurations.
        """
        mandatory_keys = ["experiment", "load_network"]
        for key in mandatory_keys:
            if key not in self.cfg_dict:
                raise ValueError(f"{key} is mandatory in the config file. Please add.")

    def save_config(self):
        """
        Saves the configuration file to the results directory.

        This method uses the `save_config` function to copy the current configuration file
        (`config.yml`) to the results directory (`result_dir`). It consolidates all experiment
        data and configuration in one location. The method checks if the network is loaded
        by referring to the `load_network` flag in the configuration dictionary.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            This method does not return any value. It performs the file copy operation.
        """
        save_config(
            self.path_to_config,
            self.result_dir,
            load_network=self.cfg_dict["load_network"],
        )
