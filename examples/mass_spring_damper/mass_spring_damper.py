# default packages
import logging
import os

# third party packages
import tensorflow as tf

# own packages
import aphin.utils.visualizations as aphin_vis
from aphin.identification import PHIN
from aphin.layers import PHLayer
from aphin.utils.data import Dataset, PHIdentifiedDataset
from aphin.utils.callbacks_tensorflow import callbacks
from aphin.utils.configuration import Configuration
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
    save_evaluation_times,
    save_training_times,
)
from aphin.utils.print_matrices import print_matrices


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
    msd_cfg = configuration.cfg_dict
    data_dir, log_dir, weight_dir, result_dir = configuration.directories

    # set up matplotlib
    aphin_vis.setup_matplotlib(msd_cfg["setup_matplotlib"])

    # Reproducibility
    # tf.config.run_functions_eagerly(True)
    tf.keras.utils.set_random_seed(msd_cfg["seed"])
    tf.config.experimental.enable_op_determinism()

    # %% Data
    logging.info(
        "################################   1. Data ################################"
    )
    # define example paths
    cache_path = os.path.join(data_dir, msd_cfg["sim_name"])

    # %% Load data
    try:
        msd_data = Dataset.from_data(cache_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File could not be found. If this is the first time you run this example, please execute the data generating script `./state_space_ph/mass_spring_damper_data_gen.py` first."
        )

    # split into train and test data
    msd_data.train_test_split(test_size=msd_cfg["test_size"], seed=msd_cfg["seed"])
    # scale data
    msd_data.scale_Mu(desired_bounds=msd_cfg["desired_bounds"])
    msd_data.states_to_features()

    t, x, dx_dt, u, mu = msd_data.data
    n_sim, n_t, n_n, n_dn, n_u, n_mu = msd_data.shape
    n_f = n_n * n_dn

    # %% Create PHAutoencoder
    logging.info(
        "################################   2. Model      ################################"
    )

    # ph identification network (pHIN)
    callback = callbacks(
        weight_dir,
        tensorboard=msd_cfg["tensorboard"],
        log_dir=log_dir,
        monitor="val_loss",
        earlystopping=True,
        patience=500,
    )

    # %% Create PHAutoencoder
    logging.info(
        "################################   2. Model      ################################"
    )

    regularizer = tf.keras.regularizers.L1L2(l1=msd_cfg["l1"], l2=msd_cfg["l2"])
    system_layer = PHLayer(
        n_f,
        n_u=n_u,
        n_mu=n_mu,
        regularizer=regularizer,
        name="ph_layer",
        layer_sizes=msd_cfg["layer_sizes_ph"],
        activation=msd_cfg["activation_ph"],
    )

    phin = PHIN(n_f, x=x, u=u, mu=mu, system_layer=system_layer, name="phin")

    #  create model with several inputs
    phin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=msd_cfg["lr"]),
        loss=tf.keras.losses.MSE,
    )

    # phin.load_weights(data_path_weights_filename)
    if msd_cfg["load_network"]:
        logging.info(f"Loading NN weights.")
        phin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    else:
        n_train = int(0.8 * x.shape[0])
        x_train = [x[:n_train], dx_dt[:n_train], u[:n_train], mu[:n_train]]
        x_val = [x[n_train:], dx_dt[n_train:], u[n_train:], mu[n_train:]]
        logging.info(f"Fitting NN weights.")
        train_hist = phin.fit(
            x=x_train,
            validation_data=(x_val, None),
            epochs=msd_cfg["n_epochs"],
            batch_size=msd_cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        save_training_times(train_hist, result_dir)
        aphin_vis.plot_train_history(train_hist)
        phin.load_weights(os.path.join(weight_dir, ".weights.h5"))

    # write data to results directory
    save_weights(weight_dir, result_dir, load_network=msd_cfg["load_network"])
    write_to_experiment_overview(
        msd_cfg, result_dir, load_network=msd_cfg["load_network"]
    )

    # calculate all needed identified results
    msd_data_id = PHIdentifiedDataset.from_identification(msd_data, system_layer, phin)

    _, _, _, _, mu_test = msd_data.test_data
    n_t_test = msd_data.n_t_test
    print_matrices(system_layer, mu=mu_test, n_t=n_t_test, data=msd_data)
    save_evaluation_times(msd_data_id, result_dir)

    msd_data.calculate_errors(msd_data_id, domain_split_vals=[1, 1])
    use_train_data = False
    aphin_vis.plot_errors(
        msd_data,
        use_train_data,
        save_name=os.path.join(result_dir, "rms_error"),
    )

    msd_data.calculate_errors(msd_data_id, save_to_txt=True, result_dir=result_dir)

    # plot state values
    idx_gen = "rand"
    aphin_vis.plot_time_trajectories_all(
        msd_data, msd_data_id, use_train_data, idx_gen, result_dir
    )

    # plot chessboard visualisation
    test_ids = [0, 1, 3, 6, 7]  # test_ids = range(10) # range(6) test_ids = [0]
    aphin_vis.chessboard_visualisation(test_ids, system_layer, msd_data, result_dir)


if __name__ == "__main__":
    main()
