# default packages
import logging
import os

# third party packages
import tensorflow as tf

# own packages
import phdl.utils.visualizations as phdl_vis
from phdl.utils.data import Dataset, PHIdentifiedDataset
from phdl.identification import APHIN, PHIN
from phdl.layers import PHQLayer
from phdl.utils.configuration import Configuration
from phdl.utils.callbacks_tensorflow import callbacks
from phdl.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
)
from phdl.utils.print_matrices import print_matrices


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
    phdl_vis.setup_matplotlib(pd_cfg["setup_matplotlib"])

    # Reproducibility
    tf.keras.utils.set_random_seed(pd_cfg["seed"])
    tf.config.experimental.enable_op_determinism()

    # %% Script parameters
    experiment = pd_cfg["experiment"]

    # %% Load simulation data
    logging.info(
        "################################   1. Data       ################################"
    )

    # %% load data
    r = pd_cfg[experiment]["r"]
    n_f = pd_cfg["n_n"] * pd_cfg["n_dn"]
    cache_path = os.path.join(data_dir, "pendulum.npz")
    pendulum_data = Dataset.from_data(cache_path)
    pendulum_data.train_test_split(test_size=0.333, seed=pd_cfg["seed"])
    pendulum_data.truncate_time(trunc_time_ratio=pd_cfg["trunc_time_ratio"])
    pendulum_data.states_to_features()

    t, x, dx_dt, _, _ = pendulum_data.data
    t_test, x_test, dx_dt_test, _, _ = pendulum_data.test_data

    # %% Create APHIN
    logging.info(
        "################################   2. Model      ################################"
    )
    if experiment == "phin":
        use_autoencoder = False
    elif experiment == "aphin_nonlinear" or experiment == "aphin_linear":
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
            use_pca=pd_cfg[experiment]["use_pca"],
            pca_only=pd_cfg[experiment]["pca_only"],
            pca_order=pd_cfg["n_pca"],
        )
    else:
        # pHIN as a subcategory of ApHIN
        aphin = PHIN(reduced_order=n_f, x=x, u=None, mu=None, system_layer=system_layer)

    aphin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=pd_cfg["lr"]),
        loss=tf.keras.losses.MSE,
    )

    callback = callbacks(
        weight_dir,
        tensorboard=pd_cfg["tensorboard"],
        log_dir=log_dir,
        monitor="loss",
        earlystopping=True,
        patience=500,
        lr_scheduler=True,
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
        phdl_vis.plot_train_history(train_hist)
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

    if experiment.startswith("aphin"):
        file_name = "projection_error.txt"
        projection_error_file_dir = os.path.join(result_dir, file_name)
        aphin.get_projection_properties(x, x_test, file_dir=projection_error_file_dir)

    pendulum_data.calculate_errors(
        pendulum_data_id,
        domain_split_vals=[2, 2],
        save_to_txt=True,
        result_dir=result_dir,
    )

    phdl_vis.plot_errors(
        pendulum_data,
        t=pendulum_data.t_test,
        save_name=os.path.join(result_dir, "rms_error"),
        domain_names=["disp", "vel"],
        save_to_csv=True,
        yscale="log",
    )

    use_train_data = False
    idx_gen = "first"
    phdl_vis.plot_time_trajectories_all(
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
        result_dir, second_oder=True, filename=experiment
    )

    if r != n_f:
        pendulum_data_id.TEST.save_latent_traj_as_csv(result_dir)


if __name__ == "__main__":
    main()
