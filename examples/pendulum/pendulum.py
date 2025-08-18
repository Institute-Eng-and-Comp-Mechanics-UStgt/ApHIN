# default packages
import logging
import os

# third party packages
import tensorflow as tf
import matplotlib.pyplot as plt

# own packages
import aphin.utils.visualizations as aphin_vis
from aphin.utils.data import Dataset, PHIdentifiedDataset
from aphin.identification import APHIN, PHIN
from aphin.layers import PHQLayer
from aphin.utils.configuration import Configuration
from aphin.utils.callbacks_tensorflow import callbacks
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
    save_training_times,
    save_evaluation_times,
)
from aphin.utils.print_matrices import print_matrices
from aphin.utils.experiments import run_various_experiments


def main(config_path_to_file=None):
    # {None} if no config file shall be loaded, else create str with path to config file
    # %% Configuration
    logging.info(f"Loading configuration")
    # Priority 1: config_path_to_file (input of main function)
    # Priority 2: manual_results_folder (below)
    # -> variable is written to config_info which is interpreted as follows (see also doc of Configuration class):
    # config_info:          -{None}                 use default config.yml that should be on the working_dir level
    #                                                 -> config for identifaction
    #                                                 -> if load_network -> load config and weights from default path
    #                         -config_filename.yml    absolute path to config file ending with .yml
    #                                                 -> config for identifaction
    #                                                 -> if load_network -> load config and weights from .yml path
    #                         -/folder/name/          absolute path of directory that includes a config.yml and .weights.h5
    #                                                 -> config for loading results
    #                         -result_folder_name     searches for a subfolder with result_folder_name under working dir that
    #                                                 includes a config.yml and .weights.h5
    #                                                 -> config for loading results
    manual_results_folder = None  # {None} if no results shall be loaded, else create str with folder name or path to results folder

    # write to config_info
    if config_path_to_file is not None:
        config_info = config_path_to_file
    elif manual_results_folder is not None:
        config_info = manual_results_folder
    else:
        config_info = None

    # set up logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # setup experiment based on config file
    working_dir = os.path.dirname(__file__)
    configuration = Configuration(working_dir, config_info)
    pd_cfg = configuration.cfg_dict
    data_dir, log_dir, weight_dir, result_dir = configuration.directories

    # set up matplotlib
    aphin_vis.setup_matplotlib(save_plots=pd_cfg["save_plots"])

    # Reproducibility
    tf.keras.utils.set_random_seed(pd_cfg["seed"])
    tf.config.experimental.enable_op_determinism()

    # %% Script parameters
    model = pd_cfg["model"]

    # %% Load simulation data
    logging.info(
        "################################   1. Data       ################################"
    )

    # %% load data
    r = pd_cfg[model]["r"]
    n_f = pd_cfg["n_n"] * pd_cfg["n_dn"]
    cache_path = os.path.join(data_dir, "pendulum.npz")
    try:
        pendulum_data = Dataset.from_data(cache_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File could not be found. If this is the first time you run this example, please execute the data generating script `./pendulum_data_generation.py` first."
        )
    pendulum_data.train_test_split(test_size=0.333, seed=pd_cfg["seed"])
    pendulum_data.truncate_time(trunc_time_ratio=pd_cfg["trunc_time_ratio"])
    pendulum_data.states_to_features()

    t, x, dx_dt, _, _ = pendulum_data.data
    t_test, x_test, dx_dt_test, _, _ = pendulum_data.test_data

    # %% Create APHIN
    logging.info(
        "################################   2. Model      ################################"
    )
    if model == "phin":
        use_autoencoder = False
    elif model == "aphin_nonlinear" or model == "aphin_linear":
        use_autoencoder = True

    regularizer = tf.keras.regularizers.L1L2(l1=pd_cfg["l1"], l2=pd_cfg["l2"])

    system_layer = PHQLayer(r if use_autoencoder else n_f, regularizer=regularizer)

    if use_autoencoder:
        aphin = APHIN(
            r,
            x=x,
            u=None,
            mu=None,
            system_layer=system_layer,
            layer_sizes=pd_cfg["layer_sizes"],
            activation=pd_cfg["activation"],
            l_rec=pd_cfg["l_rec"],
            l_dz=pd_cfg["l_dz"],
            l_dx=pd_cfg["l_dx"],
            use_pca=pd_cfg[model]["use_pca"],
            pca_only=pd_cfg[model]["pca_only"],
            pca_order=pd_cfg["n_pca"],
        )
    else:
        # pHIN as a subcategory of ApHIN
        aphin = PHIN(reduced_order=n_f, x=x, u=None, mu=None, system_layer=system_layer)

    aphin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=pd_cfg["lr"]),
        loss=tf.keras.losses.MSE,
    )
    aphin.build(input_shape=([x.shape, dx_dt.shape], None))

    callback = callbacks(
        weight_dir,
        tensorboard=pd_cfg["tensorboard"],
        log_dir=log_dir,
        monitor="loss",
        earlystopping=True,
        patience=500,
    )
    if pd_cfg["load_network"]:
        logging.info(f"Loading NN weights.")
        aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    else:
        # Fit model
        logging.info(f"Fitting NN weights.")
        train_hist = aphin.fit(
            x=[x, dx_dt],
            validation_data=([x, dx_dt], None),
            epochs=pd_cfg["n_epochs"],
            batch_size=pd_cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        save_training_times(train_hist, result_dir)
        aphin_vis.plot_train_history(train_hist)
        aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))

    # write data to results directory
    save_weights(weight_dir, result_dir, load_network=pd_cfg["load_network"])
    write_to_experiment_overview(
        pd_cfg, result_dir, load_network=pd_cfg["load_network"]
    )

    logging.info(
        "################################   3. Validation ################################"
    )

    # system_layer.print()
    pendulum_data_id = PHIdentifiedDataset.from_identification(
        pendulum_data, system_layer, aphin
    )

    print_matrices(system_layer)
    save_evaluation_times(pendulum_data_id, result_dir)

    if model.startswith("aphin"):
        file_name = "projection_error.txt"
        projection_error_file_dir = os.path.join(result_dir, file_name)
        aphin.get_projection_properties(x, x_test, file_dir=projection_error_file_dir)

    pendulum_data.calculate_errors(
        pendulum_data_id,
        domain_split_vals=[2, 2],
        save_to_txt=True,
        result_dir=result_dir,
    )

    aphin_vis.plot_errors(
        pendulum_data,
        save_name=os.path.join(result_dir, "rms_error"),
        domain_names=["disp", "vel"],
        save_to_csv=True,
        yscale="log",
    )

    use_train_data = False
    idx_gen = "first"
    plt.show()
    aphin_vis.plot_time_trajectories_all(
        pendulum_data,
        pendulum_data_id,
        use_train_data=use_train_data,
        idx_gen=idx_gen,
        result_dir=result_dir,
    )

    # reference data
    pendulum_data.TEST.save_state_traj_as_csv(
        result_dir, second_oder=True, filename="reference"
    )
    # identified data
    pendulum_data_id.TEST.save_state_traj_as_csv(
        result_dir, second_oder=True, filename=model
    )

    if r != n_f:
        pendulum_data_id.TEST.save_latent_traj_as_csv(result_dir)

    # avoid that the script stops and keep the plots open
    plt.show()


def create_variation_of_parameters():
    """
    Create a dictionary with variations of parameters for different experiments.
    """
    parameter_variation_dict = {"model": ["phin", "aphin_linear", "aphin_nonlinear"]}
    return parameter_variation_dict

if __name__ == "__main__":
    working_dir = os.path.dirname(__file__)
    calc_various_experiments = True
    if calc_various_experiments:
        logging.info(f"Multiple simulation runs...")
        # Run multiple simulation runs defined by parameter_variavation_dict
        configuration = Configuration(working_dir)
        _, log_dir, _, result_dir = configuration.directories

        run_various_experiments(
            experiment_main_script=main,  # main without parentheses
            parameter_variation_dict=create_variation_of_parameters(),
            basis_config_yml_path=os.path.join(os.path.dirname(__file__), "config.yml"),
            result_dir=result_dir,
            log_dir=log_dir,
        )
    else:
        main()
