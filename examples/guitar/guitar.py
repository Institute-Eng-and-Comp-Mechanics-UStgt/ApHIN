import logging
import os
import numpy as np
import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
import h5py
# plotly plot of random state and it's derivative
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.io as pio

from aphin.utils.data.dataset import SynRMDataset, PHIdentifiedDataset, Dataset
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis
from aphin.utils.callbacks_tensorflow import callbacks
from aphin.identification import APHIN
from aphin.layers.phq_layer import PHQLayer, PHLayer
from aphin.utils.experiments import run_various_experiments
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
    save_evaluation_times,
    save_training_times,
)
from aphin.utils.print_matrices import print_matrices

import matplotlib.pyplot as plt


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
    cfg = configuration.cfg_dict
    data_dir, log_dir, weight_dir, result_dir = configuration.directories

    aphin_vis.setup_matplotlib(cfg["setup_matplotlib"])

    guitar_data = np.load("guitar.npz")

    # # load mat file using h5py
    # data = h5py.File(cfg["matfile_path"], "r")["snapshotData"]["trajectory"]
    # t = h5py.File(cfg["matfile_path"], "r")["snapshotData"]["same4all"]["t"][:]
    # freq = data["freq"][:]
    # x =  np.array([data[data["x"][i, 0]][:] for i in range(100)])
    #
    # # save data as compressed numpy file


    # np.savez_compressed("guitar.npz", x=x, t=t, freq=freq)
    # %% truncate data in time
    end_time_step = 400
    t = np.round(guitar_data["t"], decimals=4)[:end_time_step]
    u = np.array([-np.sin(2*np.pi*freq_*t) for freq_ in guitar_data["freq"]])[:,:end_time_step]
    x = guitar_data["x"][:,:end_time_step]
    dt = (t[1] - t[0])[0]
    # numerical differentiation
    # x = x[:, :, :, np.newaxis]
    from sklearn.decomposition import PCA
    # scale data per domain
    # domain_ids = [0] + [np.sum(cfg["domain_split_vals_"][:i]) for i in range(1, len(cfg["domain_split_vals_"])+1)]
    # relative_domain_length = [split_val / np.sum(cfg["domain_split_vals_"]) * 2 for split_val in cfg["domain_split_vals_"]]
    # for i in range(len(domain_ids)-1):
    #     print(f"Scale domain {i}: {domain_ids[i]} - {domain_ids[i+1]}")
    #     print(np.abs(x[:, :, domain_ids[i]:domain_ids[i+1]]).max())
    #     x[:, :, domain_ids[i]:domain_ids[i+1]] = x[:, :, domain_ids[i]:domain_ids[i+1]]/cfg["scaling_values"][i]*relative_domain_length[i]
    # x = x[:, :, :, np.newaxis]
    # x_dt = np.gradient(x, dt, axis=1)
    # data = Dataset(t=t, X=x, X_dt=x_dt, U=u)

    # %% Individual PCA for each domain
    n_sims, n_t = x.shape[0], x.shape[1]
    from sklearn.decomposition import PCA
    # scale data per domain
    domain_ids = [0] + [np.sum(cfg["domain_split_vals_"][:i]) for i in range(1, len(cfg["domain_split_vals_"])+1)]
    # perform domain specific pca
    x_pca = []
    pcas = []
    for i in range(len(domain_ids)-1):
        print(f"Scale domain {i}: {domain_ids[i]} - {domain_ids[i+1]}")
        x_ = x[:, :, domain_ids[i]:domain_ids[i+1]]
        # todo: train test split
        x_reshaped = x_.reshape(n_sims*n_t, -1)
        pca_ = PCA(n_components=cfg["n_pca_per_domain"])
        x_pca_ = pca_.fit_transform(x_reshaped)
        logging.info(f"relative reconstruction error: {np.linalg.norm(x_reshaped - pca_.inverse_transform(x_pca_)) /
                                                       np.linalg.norm(x_reshaped)}")
        x_pca_ = x_pca_.reshape(n_sims, n_t, -1)
        x_pca.append(x_pca_)
        pcas.append(pca_)
    # scale values
    n_domains = [x_.shape[2] for x_ in x_pca]
    domain_ids = [0] + [np.sum(n_domains[:i]) for i in range(1, len(n_domains)+1)]
    scaling_values = [np.abs(x_).max() for x_ in x_pca]
    # x_scaled = [x_ / np.abs(x_).max(axis=(0,1))/n_domains[i] for i, x_ in enumerate(x_pca)]
    x_scaled = [x_ / scaling_values[i]/n_domains[i] for i, x_ in enumerate(x_pca)]
    x = np.concatenate(x_scaled, axis=2)[..., np.newaxis]
    x_dt = np.gradient(x, dt, axis=1)
    data = Dataset(t=t, X=x, X_dt=x_dt, U=u)

    # for i in range(25):
    #     plt.plot(x[0,:,i,0])
    #     plt.show()
    #
    # plt.plot(x[:, :, :, 0].max(axis=2))
    # filter data with savgol filter
    if cfg["filter_data"]:
        logging.info("Data is filtered")
        data.filter_data(interp_equidis_t=False)
    else:
        logging.info("Data is not filtered.")

    # reduced size
    r = cfg["r"]


    # pio.renderers.default = "browser"
    # idx = np.random.randint(0, x.shape[0])
    # x_plot = x[0, :, 0, idx]
    # x_plot_dt = x_dt[0, :, 0, idx]
    # x_plot_dt2 = x[0, :, 0, idx+2454+3170]
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=t.squeeze(), y=x_plot, mode="lines", name="x"))
    # fig.add_trace(go.Scatter(x=t.squeeze(), y=x_plot_dt, mode="lines", name="x_dt numerical"))
    # fig.add_trace(go.Scatter(x=t.squeeze(), y=x_plot_dt2, mode="lines", name="x_dt_data"))
    # fig.show()


    # train-test split
    data.train_test_split(cfg["test_size"], seed=cfg["seed"])

    # scale u manually
    data.reshape_inputs_to_features()

    # transform to feature form that is used by the deep learning
    data.states_to_features()
    t, x, dx_dt, u, mu = data.data

    # %% Create APHIN
    logging.info(
        "################################   2. Model      ################################"
    )
    validation = True
    if validation:
        monitor = "val_loss"
    else:
        monitor = "loss"

    n_sim, n_t, n_n, n_dn, n_u, n_mu = data.shape
    regularizer = tf.keras.regularizers.L1L2(l1=cfg["l1"], l2=cfg["l2"])

    errors_train = []
    errors_test = []

    weight_path = os.path.join(weight_dir, f".weights.h5")
    callback = callbacks(
        weight_path,
        tensorboard=cfg["tensorboard"],
        log_dir=log_dir,
        monitor=monitor,
        earlystopping=False,
        patience=500,
    )

    system_layer = PHLayer(
        r,
        n_u=n_u,
        n_mu=n_mu,
        name="phq_layer",
        layer_sizes=cfg["layer_sizes_ph"],
        activation=cfg["activation_ph"],
        regularizer=regularizer,
    )

    aphin = APHIN(
        r,
        x=x,
        u=u,
        system_layer=system_layer,
        layer_sizes=cfg["layer_sizes_ae"],
        activation=cfg["activation_ae"],
        l_rec=cfg["l_rec"],
        l_dz=cfg["l_dz"],
        l_dx=cfg["l_dx"],
        use_pca=cfg["use_pca"],
        pca_only=cfg["pca_only"],
        pca_order=cfg["n_pca"],
        pca_scaling=cfg["pca_scaling"],
        # dtype=tf.float64,
    )

    aphin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
        loss=tf.keras.losses.MSE,
    )
    aphin.build(input_shape=([x.shape, dx_dt.shape, u.shape], None))

    # aphin.load_weights(os.path.join(weight_dir, "good.weights.h5"))
    # aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))

    # x_rec = aphin.reconstruct(x)
    # e_rel = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    # Fit or learn neural network
    if cfg["load_network"]:
        logging.info(f"Loading NN weights.")
        aphin.load_weights(weight_path)
    else:
        logging.info(f"Fitting NN weights.")
        n_train = int(0.8 * x.shape[0])
        x_train = [x[:n_train], dx_dt[:n_train], u[:n_train]]
        x_val = [x[n_train:], dx_dt[n_train:], u[n_train:]]
        train_hist = aphin.fit(
            x=x_train,
            validation_data=x_val,
            epochs=cfg["n_epochs"],
            batch_size=cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        save_training_times(train_hist, result_dir)
        aphin_vis.plot_train_history(
            train_hist, save_name=result_dir, validation=validation
        )

        # load best weights
        aphin.load_weights(weight_path)

    # write data to results directory
    # save_weights(weight_dir, result_dir, load_network=cfg["load_network"])
    write_to_experiment_overview(
        cfg, result_dir, load_network=cfg["load_network"]
    )

    # reconstruct data and report error
    e_rec_train = np.linalg.norm(data.TRAIN.x - aphin.reconstruct(data.TRAIN.x)) / np.linalg.norm(data.TRAIN.x)
    e_rec_test = np.linalg.norm(data.TEST.x - aphin.reconstruct(data.TEST.x)) / np.linalg.norm(data.TEST.x)
    logging.info(f" r={r}:   {e_rec_train * 100:.2f}:%  / {e_rec_test * 100:.2f}:%")
    errors_train.append(e_rec_train)
    errors_test.append(e_rec_test)

    # %% Validation
    logging.info(
        "################################   3. Validation ################################"
    )
    t_test, x_test, dx_dt_test, u_test, mu_test = data.test_data
    n_sim_test, n_t_test, _, _, _, _ = data.shape_test
    print_matrices(system_layer, mu=mu_test, n_t=n_t_test)

    # calculate projection and Jacobian errors
    # file_name = "projection_error.txt"
    # projection_error_file_dir = os.path.join(result_dir, file_name)
    # aphin.get_projection_properties(x, x_test, file_dir=projection_error_file_dir)

    # %% Validation of the AE reconstruction
    # get original quantities
    data_id = PHIdentifiedDataset.from_identification(
        data, system_layer, aphin, integrator_type="imr"
    )
    save_evaluation_times(data_id, result_dir)

    # reproject
    # num_rand_pick_entries = 1000
    # data.reproject_with_basis(
    #     [V, V],
    #     idx=[slice(80, 100), slice(105, 125)],
    #     pick_method="rand",
    #     pick_entry=num_rand_pick_entries,
    #     seed=cfg["seed"],
    # )
    # data_id.reproject_with_basis(
    #     [V, V],
    #     idx=[slice(80, 100), slice(105, 125)],
    #     pick_method="rand",
    #     pick_entry=num_rand_pick_entries,
    #     seed=cfg["seed"],
    # )

    # domain_split_vals_projected = [
    #     3,
    #     72,
    #     5,
    #     num_rand_pick_entries,
    #     5,
    #     num_rand_pick_entries,
    # ]

    # data.calculate_errors(
    #     data_id,
    #     domain_split_vals=domain_split_vals_projected,
    #     save_to_txt=True,
    #     result_dir=result_dir,
    # )

    e_rec = np.linalg.norm(data.TRAIN.X - data_id.TRAIN.X_rec) / np.linalg.norm(data.TRAIN.X)
    e_z = np.linalg.norm(data_id.TRAIN.Z - data_id.TRAIN.Z_ph) / np.linalg.norm(data_id.TRAIN.Z)
    e_z_dt = np.linalg.norm(data_id.TRAIN.Z_dt - data_id.TRAIN.Z_dt_ph) / np.linalg.norm(data_id.TRAIN.Z_dt)
    e_z_dt_map = np.linalg.norm(data_id.TRAIN.Z_dt - data_id.TRAIN.Z_dt_ph_map) / np.linalg.norm(data_id.TRAIN.Z_dt)
    e_x = np.linalg.norm(data.TRAIN.X - data_id.TRAIN.X) / np.linalg.norm(data.TRAIN.X)

    e_rec = np.linalg.norm(data.TEST.X - data_id.TEST.X_rec) / np.linalg.norm(data.TEST.X)
    e_z = np.linalg.norm(data_id.TEST.Z - data_id.TEST.Z_ph) / np.linalg.norm(data_id.TEST.Z)
    e_z_dt = np.linalg.norm(data_id.TEST.Z_dt - data_id.TEST.Z_dt_ph) / np.linalg.norm(data_id.TEST.Z_dt)
    e_z_dt_map = np.linalg.norm(data_id.TEST.Z_dt - data_id.TEST.Z_dt_ph_map) / np.linalg.norm(data_id.TEST.Z_dt)
    e_x = np.linalg.norm(data.TEST.X - data_id.TEST.X) / np.linalg.norm(data.TEST.X)

    # %% Inverse PCA transformation
    from sklearn.model_selection import train_test_split
    X_ref = guitar_data["x"]
    X_ref_train, X_ref_test =  train_test_split(X_ref, test_size=cfg["test_size"], random_state=cfg["seed"])
    X = []
    for i in range(len(domain_ids)-1):
        x_id = data_id.TEST.X[:, :, domain_ids[i]:domain_ids[i+1]]
        x_id = x_id * scaling_values[i] * n_domains[i]
        x_id = pcas[i].inverse_transform(x_id.reshape(n_sim_test*n_t_test, -1)).reshape(n_sim_test, n_t_test, -1)
        X.append(x_id)

    X = np.concatenate(X, axis=2)
    e_rel_X = np.linalg.norm(X_ref_test - X) / np.linalg.norm(X_ref_test)
    logging.info(f"Relative error on the inverse PCA transformation: {e_rel_X:.2f}%")



    #
    # for i in range(16):
    #     plt.plot(data_id.TEST.Z_ph[0, :, i])
    #     plt.plot(data_id.TEST.Z[0, :, i])
    #     plt.show()
    #
    # x_rec_ = aphin.decode(data_id.TEST.Z_ph[0])
    # for i in range(16):
    #     plt.plot(x_rec_[:, i])
    #     plt.plot(data.TEST.X[0, :, i, 0])
    #     plt.show()
    #
    # e_x_mech = (np.linalg.norm(data.TEST.X[:,:,:domain_ids[1]] - data_id.TEST.X[:,:,:domain_ids[1]]) /
    #        np.linalg.norm(data.TEST.X[:,:,:domain_ids[1]]))
    #
    # e_x_acc = (np.linalg.norm(data.TEST.X[:,:,domain_ids[1]:domain_ids[2]] - data_id.TEST.X[:,:,domain_ids[1]:domain_ids[2]]) /
    #        np.linalg.norm(data.TEST.X[:,:,domain_ids[1]:domain_ids[2]]))
    #
    # e_x_Dmech = (np.linalg.norm(data.TEST.X[:,:,domain_ids[2]:domain_ids[3]] - data_id.TEST.X[:,:,domain_ids[2]:domain_ids[3]]) /
    #           np.linalg.norm(data.TEST.X[:,:,domain_ids[2]:domain_ids[3]]))
    #
    # e_x_Dacc = (np.linalg.norm(data.TEST.X[:,:,domain_ids[3]:domain_ids[4]] - data_id.TEST.X[:,:,domain_ids[3]:domain_ids[4]]) /
    #             np.linalg.norm(data.TEST.X[:,:,domain_ids[3]:domain_ids[4]]))

    # i = 0
    # plt.plot(data_id.TEST.Z_dt[0, :, i])
    # # plt.plot(data_id.TEST.Z_dt_ph[0, :, i])
    # plt.plot(data_id.TEST.Z_dt_ph_map[0, :, i])
    # # plt.legend(["Z_dt", "Z_dt_ph", "Z_dt_ph_map"])
    # plt.show()

    # from aphin.utils.visualizations import plot_Z_dt_ph, plot_Z_dt_ph_map, plot_Z_ph
    #
    #
    # plot_Z_dt_ph(
    #         data_id,
    #         use_train_data=False,
    #         idx_gen="first",
    #         save_path=result_dir,
    #         idx_custom_tuple= [range(3), range(4)],
    #     )
    # plot_Z_dt_ph_map(
    #         data_id,
    #         use_train_data=False,
    #         idx_gen="first",
    #         save_path=result_dir,
    #         idx_custom_tuple= [range(3), range(4)],
    #     )
    #
    # plot_Z_ph(
    #         data_id,
    #         use_train_data=False,
    #         idx_gen="first",
    #         save_path=result_dir,
    #         idx_custom_tuple= [range(3), range(4)],
    # )

    print(f"Relative error on the reconstruction: {e_rec * 100:.2f}%")
    print(f"Relative error on the latent variable: {e_z * 100:.2f}%")
    print(f"Relative error on the latent variable time derivative: {e_z_dt * 100:.2f}%")
    print(f"Relative error on the latent variable time derivative map: {e_z_dt_map * 100:.2f}%")
    print(f"Relative error on the state variable: {e_x * 100:.2f}%")

    # %% calculate errors
    data.calculate_errors(
        data_id,
        domain_split_vals=cfg["domain_split_vals"],
        save_to_txt=True,
        result_dir=result_dir,
    )
    aphin_vis.plot_errors(
        data,
        t=data.t_test,
        save_name=os.path.join(result_dir, "rms_error"),
        domain_names=cfg["domain_names"],
        save_to_csv=True,
        yscale="log",
    )


    # %% plot trajectories
    use_train_data = False
    use_rand = True
    if use_rand:
        idx_gen = "rand"
        idx_custom_tuple = None
    else:
        # use predefined indices
        idx_gen = "custom"
        # idx_eta = np.arange(3)
        # idx_phi = np.arange(3, 75)
        # idx_rigid = np.arange(75, 80)
        # idx_elastic_modes = np.arange(80, 100)
        # idx_Drigid = np.arange(100, 105)
        # idx_Delastic = np.arange(105, 125)
        # all domains
        # idx_n_n = np.array([0] * 7)
        # idx_n_dn = np.array([0, 3, 4, 75, 80, 101, 106])
        # no velocities
        idx_n_n = np.array([0] * 5)
        idx_n_dn = np.array([0, 3, 4, 75, 80])
        idx_n_f = np.array([0, 4, 13, 20, 25])  # for latent space
        idx_custom_tuple = [
            (idx_n_n[i], idx_n_dn[i], idx_n_f[i]) for i in range(idx_n_n.shape[0])
        ]
    aphin_vis.plot_time_trajectories_all(
        data,
        data_id,
        use_train_data=use_train_data,
        idx_gen=idx_gen,
        result_dir=result_dir,
        idx_custom_tuple=idx_custom_tuple,
    )

    aphin_vis.plot_u(data=data, use_train_data=use_train_data)

    # avoid that the script stops and keep the plots open
    plt.show()

    print("debug")

def create_variation_of_parameters():
    parameter_variation_dict = {
        "r": [8, 16, 24],
    }
    return parameter_variation_dict

if __name__ == "__main__":
    calc_various_experiments = False
    if calc_various_experiments:
        logging.info(f"Multiple simulation runs...")
        # Run multiple simulation runs defined by parameter_variavation_dict
        working_dir = os.path.dirname(__file__)
        configuration = Configuration(working_dir)
        _, log_dir, _, result_dir = configuration.directories
        run_various_experiments(
            experiment_main_script=main,  # main without parentheses
            parameter_variation_dict=create_variation_of_parameters(),
            basis_config_yml_path=os.path.join(os.path.dirname(__file__), "config.yml"),
            result_dir=result_dir,
            log_dir=log_dir,
            force_calculation=False,
        )
    else:
        main()
