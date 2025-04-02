# default packages
import logging
import os

import numpy as np
# third party packages
import tensorflow as tf
# import matplotlib
# matplotlib.use("TkAgg")  # Force interactive backend
import matplotlib.pyplot as plt

# own packages
import aphin.utils.visualizations as aphin_vis
from aphin.identification import PHIN
from aphin.layers import PHLayer, LTILayer
from aphin.utils.data import Dataset, PHIdentifiedDataset
from aphin.utils.callbacks_tensorflow import callbacks
from aphin.utils.configuration import Configuration
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
    save_evaluation_times,
    save_training_times,
)
from aphin.utils.visualizations import save_as_png
from aphin.utils.print_matrices import print_matrices

# tf.config.run_functions_eagerly(True)
#
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
    aphin_vis.setup_matplotlib(msd_cfg["save_plots"])

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
            f"File could not be found. If this is the first time you run this example, please execute the data generating script `./data_generation/mass_spring_damper_data_gen.py` first."
        )

    # split into train and test data
    msd_data.train_test_split(test_size=msd_cfg["test_size"], seed=msd_cfg["seed"])
    msd_data.truncate_time(trunc_time_ratio=msd_cfg["trunc_time_ratio"])

    # scale data
    msd_data.scale_Mu(desired_bounds=msd_cfg["desired_bounds"])
    msd_data.states_to_features()

    t, x, dx_dt, u, mu = msd_data.data
    n_sim, n_t, n_n, n_dn, n_u, n_mu = msd_data.shape
    n_f = n_n * n_dn

    # plt.figure()
    # plt.plot(t, msd_data.Data[0][0, :, 0, 0])
    # plt.plot(msd_data.t_test, msd_data.Data_test[0][0, :, 0, 0])
    # plt.show()

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

    if "ph" in msd_cfg["experiment"]:
        system_layer = PHLayer(
            n_f,
            n_u=n_u,
            n_mu=n_mu,
            regularizer=regularizer,
            name="ph_layer",
            layer_sizes=msd_cfg["layer_sizes_ph"],
            activation=msd_cfg["activation_ph"],
        )
    elif "lti" in msd_cfg["experiment"]:
        system_layer = LTILayer(
            n_f,
            n_u=n_u,
            n_mu=n_mu,
            regularizer=regularizer,
            name="lti_layer",
            layer_sizes=msd_cfg["layer_sizes_ph"],
            activation=msd_cfg["activation_ph"],
        )

    phin = PHIN(n_f, x=x, u=u, mu=mu, system_layer=system_layer, name="phin")

    #  create model with several inputs
    phin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=msd_cfg["lr"]),
        loss=tf.keras.losses.MSE,
    )
    n_train = int(0.8 * x.shape[0])
    import numpy as np
    if np.any(u):
        x_train = [x[:n_train], dx_dt[:n_train], u[:n_train], mu[:n_train]]
        x_val = [x[n_train:], dx_dt[n_train:], u[n_train:], mu[n_train:]]
    else:
        x_train = [x[:n_train], dx_dt[:n_train], np.empty((x[:n_train].shape[0], 0)), mu[:n_train]]
        x_val = [x[n_train:], dx_dt[n_train:], np.empty((x[n_train:].shape[0], 0)), mu[n_train:]]
    phin.build(input_shape=([data_.shape for data_ in x_train], None))

    # phin.load_weights(data_path_weights_filename)
    phin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    if msd_cfg["load_network"]:
        logging.info(f"Loading NN weights.")
        phin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    else:
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
        aphin_vis.plot_train_history(train_hist, save_name=os.path.join(result_dir, "train_history"))
        aphin_vis.plot_train_history(train_hist, validation=True, save_name=os.path.join(result_dir, "train_history"))
        phin.load_weights(os.path.join(weight_dir, ".weights.h5"))

    # write data to results directory
    save_weights(weight_dir, result_dir, load_network=msd_cfg["load_network"])
    write_to_experiment_overview(
        msd_cfg, result_dir, load_network=msd_cfg["load_network"]
    )

    # calculate all needed identified results
    msd_data_id = PHIdentifiedDataset.from_identification(msd_data, system_layer, phin)

    _, _, _, _, mu_test = msd_data.test_data
    n_t_test = msd_data.n_t
    print_matrices(system_layer, mu=mu_test, n_t=n_t_test, data=msd_data)
    save_evaluation_times(msd_data_id, result_dir)


    msd_data.calculate_errors(msd_data_id, domain_split_vals=[1, 1])
    use_train_data = False
    aphin_vis.plot_errors(
        msd_data,
        use_train_data,
        save_to_csv=True,
        save_name=os.path.join(result_dir, "rms_error"),
    )

    msd_data.calculate_errors(msd_data_id, save_to_txt=True, result_dir=result_dir)

    # reference data
    for dof in range(3):
        msd_data.TEST.save_state_traj_as_csv(
            result_dir, second_oder=True, dof=dof, filename=f"state_{dof}_trajectories_reference"
        )
        # identified data
        msd_data_id.TEST.save_state_traj_as_csv(
            result_dir, second_oder=True, dof=dof, filename=f"state_{dof}_trajectories_{msd_cfg["experiment"]}"
        )

    # plot state values
    idx_gen = "rand"
    aphin_vis.plot_time_trajectories_all(
        msd_data, msd_data_id, use_train_data, idx_gen, result_dir
    )


    # predicted matrices
    J_pred, R_pred, B_pred = system_layer.get_system_matrices(mu_test, n_t=n_t_test)
    # original test matrices
    J_test_, R_test_, Q_test_, B_test_ = msd_data.ph_matrices_test

    A_pred = J_pred - R_pred
    A_test = (J_test_ - R_test_) @ Q_test_

    error_A = np.linalg.norm(A_pred - A_test, axis=(1,2)) / np.linalg.norm(A_test, axis=(1,2))

    import numpy as np
    max_eigs_pred = np.array([np.real(np.linalg.eig(A_).eigenvalues).max() for A_ in A_pred])
    max_eigs_ref = np.array([np.real(np.linalg.eig(A_).eigenvalues).max() for A_ in A_test])

    # plot eigenvalues on imaginary axis
    eigs_pred = np.array([np.linalg.eig(A_).eigenvalues for A_ in A_pred])
    eigs_ref = np.array([np.linalg.eig(A_).eigenvalues for A_ in A_test])

    # save real and imaginary parts of eigenvalues to csv
    header = "".join([f"sim{i}_eigs_real," for i in range(eigs_pred.shape[0])]) + "".join([f"sim{i}_eigs_imag," for i in range(eigs_pred.shape[0])])
    np.savetxt(os.path.join(result_dir, "eigenvalues_ref.csv"),
               np.concatenate([eigs_ref.real, eigs_ref.imag], axis=0).T, delimiter=",", header=header, comments="")

    np.savetxt(os.path.join(result_dir, "eigenvalues_pred.csv"),
               np.concatenate([eigs_pred.real, eigs_pred.imag], axis=0).T, delimiter=",", header=header, comments="")

    # close all figures
    # plt.close("all")
    es = []
    i_test = 1
    for i_test in range(msd_data_id.TEST.Z_ph.shape[0]):
        plt.figure()
        plt.plot(msd_data_id.TEST.Z_ph[i_test, :, 0], label="pred")
        plt.plot(msd_data_id.TEST.Z[i_test, :, 0], label="ref")
        plt.legend()
        e = np.linalg.norm(msd_data_id.TEST.Z_ph[i_test, :, 0] - msd_data_id.TEST.Z[i_test, :, 0]) / np.linalg.norm(msd_data_id.TEST.Z[i_test, :, 0])
        es.append(e)
        print(e)
    es = np.array(es)
    plt.show()
    print(np.array(es).mean())

    plt.figure()
    plt.plot(error_A)
    plt.plot(es)
    plt.show()


    test_ids = [0, 1, 3, 6, 7]
    for i in test_ids:
        plt.figure()
        plt.plot(eigs_pred[i].real, eigs_pred[i].imag, "x", label="pred")
        plt.plot(eigs_ref[i].real, eigs_ref[i].imag, "o", label="ref")
        plt.xlabel("real")
        plt.ylabel("imag")
        plt.legend()
        plt.show(block=False)
        save_as_png(os.path.join(result_dir, f"eigenvalues_{i}"))
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5.701, 3.5), dpi=300, sharex="all")
    plt.plot(max_eigs_pred, label="pred")
    plt.plot(max_eigs_ref, label="ref")
    plt.xlabel("test id")
    plt.ylabel("maximum eigenvalue")
    # fig.legend(loc='outside center right', bbox_to_anchor=(1.3, 0.6))
    plt.tight_layout()
    plt.show(block=False)
    save_as_png(os.path.join(result_dir, "maximum_eigenvalues"))

    # plot chessboard visualisation
    test_ids = [0, 1, 3, 6, 7]  # test_ids = range(10) # range(6) test_ids = [0]
    aphin_vis.chessboard_visualisation(test_ids, system_layer, msd_data, result_dir, error_limits=[0.022277599200606346, 0.01978847570717334, 0.04014775591542816, 0.023013601874971812])

    # avoid that the script stops and keep the plots open
    plt.show()
    print('a')

if __name__ == "__main__":
    main()
