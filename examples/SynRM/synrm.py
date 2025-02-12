import logging
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from aphin.utils.data.dataset import SynRMDataset, PHIdentifiedDataset
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis
from aphin.utils.callbacks_tensorflow import callbacks
from aphin.identification import APHIN
from aphin.layers.phq_layer import PHQLayer
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
    save_evaluation_times,
    save_training_times,
)
from aphin.utils.print_matrices import print_matrices


def main(
    config_path_to_file=None,
):  # {None} if no config file shall be loaded, else create str with path to config file
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

    # setup experiment based on config file
    working_dir = os.path.dirname(__file__)
    configuration = Configuration(working_dir, config_info)
    srm_cfg = configuration.cfg_dict
    data_dir, log_dir, weight_dir, result_dir = configuration.directories

    synrm_data = SynRMDataset.from_matlab(data_path=srm_cfg["matfile_path"])

    aphin_vis.setup_matplotlib(srm_cfg["setup_matplotlib"])

    # reduced size
    r = srm_cfg["r"]

    # train-test split
    sim_idx_train = np.arange(10)
    sim_idx_test = np.arange(5) + len(sim_idx_train)
    synrm_data.train_test_split_sim_idx(sim_idx_train, sim_idx_test)

    # filter data with savgol filter
    if srm_cfg["filter_data"]:
        logging.info("Data is filtered")
        synrm_data.filter_data(interp_equidis_t=False)
    else:
        logging.info("Data is not filtered.")

    # scale data
    # synrm_data.scale_all(
    #     scaling_values=srm_cfg["scaling_values"],
    #     domain_split_vals=srm_cfg["domain_split_vals"],
    #     u_desired_bounds=srm_cfg["desired_bounds"],
    #     mu_desired_bounds=srm_cfg["desired_bounds"],
    # )

    # transform to feature form that is used by the deep learning
    synrm_data.states_to_features()
    t, x, dx_dt, u, mu = synrm_data.data

    # %% Create APHIN
    logging.info(
        "################################   2. Model      ################################"
    )

    callback = callbacks(
        weight_dir,
        tensorboard=srm_cfg["tensorboard"],
        log_dir=log_dir,
        monitor="loss",
        earlystopping=True,
        patience=500,
    )

    n_sim, n_t, n_n, n_dn, n_u, n_mu = synrm_data.shape
    regularizer = tf.keras.regularizers.L1L2(l1=srm_cfg["l1"], l2=srm_cfg["l2"])
    system_layer = PHQLayer(
        r,
        n_u=n_u,
        n_mu=n_mu,
        name="phq_layer",
        layer_sizes=srm_cfg["layer_sizes_ph"],
        activation=srm_cfg["activation_ph"],
        regularizer=regularizer,
        # dtype=tf.float64,
    )

    aphin = APHIN(
        r,
        x=x,
        u=u,
        system_layer=system_layer,
        layer_sizes=srm_cfg["layer_sizes_ae"],
        activation=srm_cfg["activation_ae"],
        l_rec=srm_cfg["l_rec"],
        l_dz=srm_cfg["l_dz"],
        l_dx=srm_cfg["l_dx"],
        use_pca=srm_cfg["use_pca"],
        pca_only=srm_cfg["pca_only"],
        pca_order=srm_cfg["n_pca"],
        pca_scaling=srm_cfg["pca_scaling"],
        # dtype=tf.float64,
    )

    aphin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=srm_cfg["lr"]),
        loss=tf.keras.losses.MSE,
    )
    aphin.build(input_shape=([x.shape, dx_dt.shape, u.shape], None))

    # Fit or learn neural network
    if srm_cfg["load_network"]:
        logging.info(f"Loading NN weights.")
        aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    else:
        logging.info(f"Fitting NN weights.")
        train_hist = aphin.fit(
            x=[x, dx_dt, u],
            # validation_data=([x, u_x, dx_dt], None),
            epochs=srm_cfg["n_epochs"],
            batch_size=srm_cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        save_training_times(train_hist, result_dir)
        aphin_vis.plot_train_history(train_hist)

        # load best weights
        aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))

        # %% Validation
    logging.info(
        "################################   3. Validation ################################"
    )
    t_test, x_test, dx_dt_test, u_test, mu_test = synrm_data.test_data
    n_sim_test, n_t_test, _, _, _, _ = synrm_data.shape_test
    print_matrices(system_layer, mu=mu_test, n_t=n_t_test)

    # calculate projection and Jacobian errors
    file_name = "projection_error.txt"
    projection_error_file_dir = os.path.join(result_dir, file_name)
    aphin.get_projection_properties(x, x_test, file_dir=projection_error_file_dir)

    # %% Validation of the AE reconstruction
    # get original quantities
    synrm_data_id = PHIdentifiedDataset.from_identification(
        synrm_data, system_layer, aphin, integrator_type="imr"
    )
    save_evaluation_times(synrm_data_id, result_dir)

    # %% calculate errors
    synrm_data.calculate_errors(
        synrm_data_id,
        # domain_split_vals=srm_cfg["domain_split_vals"],
        save_to_txt=True,
        result_dir=result_dir,
    )
    aphin_vis.plot_errors(
        synrm_data,
        t=synrm_data.t_test,
        save_name=os.path.join(result_dir, "rms_error"),
        # domain_names=srm_cfg["domain_names"],
        save_to_csv=True,
        yscale="log",
    )

    # %% plot trajectories
    use_train_data = False
    idx_gen = "rand"
    aphin_vis.plot_time_trajectories_all(
        synrm_data,
        synrm_data_id,
        use_train_data=use_train_data,
        idx_gen=idx_gen,
        result_dir=result_dir,
    )

    # avoid that the script stops and keep the plots open
    plt.show()

    print("debug")


if __name__ == "__main__":
    main()
