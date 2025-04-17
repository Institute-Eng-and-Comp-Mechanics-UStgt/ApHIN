# default packages
import numpy as np
import logging
import os
import scipy

# third party packages
import tensorflow as tf
import matplotlib.pyplot as plt

# own packages
from visualizer import Visualizer
from aphin.utils.data import PHIdentifiedDataset, DiscBrakeDataset
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
    save_evaluation_times,
    save_training_times,
    save_config_sweep_data,
)
from aphin.identification import APHIN
from aphin.layers.phq_layer import PHQLayer, PHLayer
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis
from aphin.utils.experiments import run_various_experiments
from aphin.utils.callbacks_tensorflow import callbacks
from aphin.utils.print_matrices import print_matrices
import urllib.request
import copy


# tf.config.run_functions_eagerly(True)


# %% Configuration
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
    manual_results_folder = "db_large_heat_larger_data_epoch6000_rsweep_multExp_r12"  # {None} if no results shall be loaded, else create str with folder name or path to results folder

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
    db_cfg = configuration.cfg_dict
    data_dir, log_dir, weight_dir, result_dir = configuration.directories

    # default config values
    if not "cut_time_start_and_end" in db_cfg.keys():
        db_cfg["cut_time_start_and_end"] = False
    if not "save_many_weights" in db_cfg.keys():
        db_cfg["save_many_weights"] = False
    if "layer" in db_cfg.keys():
        db_cfg["ph_layer"] = db_cfg["layer"]
    if not "ph_layer" in db_cfg.keys():
        db_cfg["ph_layer"] = "phq"
    if not "only_save" in db_cfg.keys():
        db_cfg["only_save"] = False
    if not "ref_coords" in db_cfg.keys():
        db_cfg["ref_coords"] = "disc_nodes.npy"

    # save_config_sweep_data(
    #     root_result_dir=os.path.dirname(result_dir),
    #     common_folder_name="db_large_heat_larger_data_epoch6000_rsweep_multExp",
    #     sweep_key="r",
    #     domain_names=db_cfg["domain_names"],
    # )

    aphin_vis.setup_matplotlib(db_cfg["setup_matplotlib"])
    # %% Data
    logging.info(
        "################################   1. Data ################################"
    )
    # save/load data path
    sim_name = db_cfg["sim_name"]
    cache_path = os.path.join(data_dir, f"{sim_name}.npz")  # path to .npz file

    # check if data already exists on local machine
    # if not os.path.isfile(cache_path):
    #     # download file if it is missing
    #     file_url = db_cfg["file_url"]
    #     logging.info(f"download data from {file_url} and save it to {cache_path}.")
    #     urllib.request.urlretrieve(
    #         file_url,
    #         cache_path,
    #     )

    # reduced size
    r = db_cfg["r"]

    # %% load/create data
    # Initialize Dataset
    disc_brake_data = DiscBrakeDataset.from_data(
        cache_path,
        use_velocities=db_cfg["use_velocities"],
        use_savgol=db_cfg["use_savgol"],
        num_time_steps=db_cfg["num_time_steps_load"],
    )

    # filter data with savgol filter
    if db_cfg["filter_data"]:
        logging.info("Data is filtered")
        disc_brake_data.filter_data(
            interp_equidis_t=db_cfg["interp_equidis_t"],
            window=db_cfg["window"],
            order=db_cfg["order"],
        )
    else:
        logging.info("Data is not filtered.")

    # %%
    # split into train and test data
    train_test_split_method = db_cfg["train_test_split_method"]
    if train_test_split_method == "sim_idx":
        if (
            db_cfg["sim_name"] == "disc_brake_with_hole_small_with_vel"
            or db_cfg["sim_name"]
            == "disc_brake_with_hole_small_with_vel_small_time_steps"
            or db_cfg["sim_name"]
            == "disc_brake_with_hole_without_force_small_data_large_heat"
        ):
            # small
            sim_idx_train = np.arange(10)
            sim_idx_test = np.arange(5) + len(sim_idx_train)
        elif (
            db_cfg["sim_name"] == "disc_brake_with_hole_very_small_with_vel"
            or db_cfg["sim_name"] == "disc_brake_with_hole_very_small_no_vel"
            or db_cfg["sim_name"] == "disc_brake_with_hole_very_small_with_vel_savgol"
            or db_cfg["sim_name"]
            == "disc_brake_with_hole_very_small_with_vel_small_time_steps"
        ):
            # very small
            sim_idx_train = np.arange(2)
            sim_idx_test = np.arange(1) + len(sim_idx_train)
        elif db_cfg["sim_name"] == "test_data":
            sim_idx_train = [0]
            sim_idx_test = [1]
        else:
            logging.warning(
                f"!!!!!!!!!!!!!!!!!!!!!!!! Unknown sim name. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
        disc_brake_data.train_test_split_sim_idx(sim_idx_train, sim_idx_test)
    elif train_test_split_method == "rand":
        test_size = db_cfg["test_size"]
        disc_brake_data.train_test_split(test_size=test_size, seed=db_cfg["seed"])
    else:
        raise ValueError(
            f"Unknown option for train_test_split {train_test_split_method}."
        )

    # decrease number of simulations
    if db_cfg["num_sim"] is not None:
        disc_brake_data.decrease_num_simulations(
            num_sim=db_cfg["num_sim"], seed=db_cfg["seed"]
        )

    if db_cfg["num_time_steps"] is not None:
        logging.info("Number of time steps is reduced.")
        disc_brake_data.decrease_num_time_steps(num_time_steps=db_cfg["num_time_steps"])

    # save smaller version of data for faster loading times
    # disc_brake_data.save_data_conc(
    #     data_dir=data_dir,
    #     save_name=f"disc_brake_with_hole_without_force_small_data_large_heat",
    # )

    # scale data
    # disc_brake_data.scale_all(
    #     scaling_values=db_cfg["scaling_values"],
    #     domain_split_vals=db_cfg["domain_split_vals"],
    #     u_desired_bounds=db_cfg["desired_bounds"],
    #     mu_desired_bounds=db_cfg["desired_bounds"],
    # )

    if db_cfg["cut_time_start_and_end"]:
        disc_brake_data.cut_time_start_and_end()

    disc_brake_data.scale_X(
        scaling_values=db_cfg["scaling_values"],
        domain_split_vals=db_cfg["domain_split_vals"],
    )
    if not db_cfg["sim_name"] == "test_data":
        disc_brake_data.scale_Mu(
            mu_train_bounds=None, desired_bounds=db_cfg["desired_bounds"]
        )

    # scale u manually
    u_domains = [1]
    start_idx = 0
    input_scaling_values = []
    for u_domain in u_domains:
        scaling_value = np.max(
            np.abs(disc_brake_data.TRAIN.U[:, :, start_idx : start_idx + u_domain])
        )
        disc_brake_data.TRAIN.U[:, :, start_idx : start_idx + u_domain] = (
            disc_brake_data.TRAIN.U[:, :, start_idx : start_idx + u_domain]
            / scaling_value
        )
        disc_brake_data.TEST.U[:, :, start_idx : start_idx + u_domain] = (
            disc_brake_data.TEST.U[:, :, start_idx : start_idx + u_domain]
            / scaling_value
        )
        input_scaling_values.append(scaling_value)
        start_idx += u_domain
    disc_brake_data.reshape_inputs_to_features()

    # transform to feature form that is used by the deep learning
    disc_brake_data.states_to_features()
    t, x, dx_dt, u, mu = disc_brake_data.data

    # plot state
    # num_plots = 2
    # n_sim_plot = 1
    # idx_n_n = [1300, 1400, 1500, 1842]  # nodes
    # fig, ax = aphin_vis.new_fig(num_plots, window_title="")
    # for i in range(num_plots):
    #     if i == 0:
    #         idx_n_dn = 0
    #     elif i == 1:
    #         idx_n_dn = 3
    #     ax[i].plot(t, disc_brake_data.TRAIN.X[n_sim_plot, :, idx_n_n, idx_n_dn].T)
    #     ax[i].grid(linestyle=":", linewidth=1)
    #     # ax[i].set_title(title)
    #     ax[i].legend([f"x{i_n_n}" for i_n_n in idx_n_n])
    # plt.xlabel(f"Time")
    # plt.show(block=True)

    # fig, ax = aphin_vis.new_fig(1, window_title="")
    # ax.plot(t.ravel()[:10], x[:10, 7371])
    # grad = np.gradient(x[:10, 7371], t.ravel()[:10] - t.ravel()[0])
    # x_dt_savgol = scipy.signal.savgol_filter(
    #     x[:, 7371],
    #     window_length=20,
    #     polyorder=1,
    #     deriv=1,
    #     axis=0,
    #     delta=t.ravel()[1] - t.ravel()[0],
    # )
    # ax.plot(t.ravel()[:10], grad - grad[0])
    # ax.plot(t.ravel()[:10], x_dt_savgol[:10] - x_dt_savgol[0])
    # plt.show(block=True)

    # aphin_vis.compare_x_and_x_dt(disc_brake_data, use_train_data=True, idx_gen="rand")
    # aphin_vis.plot_u(disc_brake_data, use_train_data=True)
    # %% Create APHIN
    logging.info(
        "################################   2. Model      ################################"
    )
    validation = True
    if validation:
        monitor = "val_loss"
    else:
        monitor = "loss"

    callback = callbacks(
        weight_dir,
        tensorboard=db_cfg["tensorboard"],
        log_dir=log_dir,
        monitor=monitor,
        earlystopping=True,
        patience=500,
        save_many_weights=db_cfg["save_many_weights"],
    )

    n_sim, n_t, n_n, n_dn, n_u, n_mu = disc_brake_data.shape
    regularizer = tf.keras.regularizers.L1L2(l1=db_cfg["l1"], l2=db_cfg["l2"])

    if db_cfg["ph_layer"] == "phq":
        system_layer = PHQLayer(
            r,
            n_u=n_u,
            n_mu=n_mu,
            name="phq_layer",
            layer_sizes=db_cfg["layer_sizes_ph"],
            activation=db_cfg["activation_ph"],
            regularizer=regularizer,
            # dtype=tf.float64,
        )
    elif db_cfg["ph_layer"] == "ph":
        system_layer = PHLayer(
            r,
            n_u=n_u,
            n_mu=n_mu,
            name="ph_layer",
            layer_sizes=db_cfg["layer_sizes_ph"],
            activation=db_cfg["activation_ph"],
            regularizer=regularizer,
            # dtype=tf.float64,
        )

    aphin = APHIN(
        r,
        x=x,
        u=u,
        mu=mu,
        system_layer=system_layer,
        layer_sizes=db_cfg["layer_sizes_ae"],
        activation=db_cfg["activation_ae"],
        l_rec=db_cfg["l_rec"],
        l_dz=db_cfg["l_dz"],
        l_dx=db_cfg["l_dx"],
        use_pca=db_cfg["use_pca"],
        pca_only=db_cfg["pca_only"],
        pca_order=db_cfg["n_pca"],
        pca_scaling=db_cfg["pca_scaling"],
        # dtype=tf.float64,
    )

    aphin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=db_cfg["lr"]),
        loss=tf.keras.losses.MSE,
    )
    aphin.build(input_shape=([x.shape, dx_dt.shape, u.shape, mu.shape], None))

    # # plot coefficient evolution
    # aphin_vis.plot_weight_coefficient_evolution(
    #     aphin,
    #     disc_brake_data,
    #     result_dir,
    #     weight_dir.replace("results", "weights"),
    #     weight_name_pre_weights="before_ep",
    #     every_n_epoch=2,
    #     weight_indices=None,
    #     use_train_data=use_train_data,
    # )

    # Fit or learn neural network
    if db_cfg["load_network"]:
        logging.info(f"Loading NN weights.")
        aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    else:
        logging.info(f"Fitting NN weights.")
        if validation:
            n_train = int(0.8 * x.shape[0])
            x_train = [x[:n_train], dx_dt[:n_train], u[:n_train], mu[:n_train]]
            x_val = [x[n_train:], dx_dt[n_train:], u[n_train:], mu[n_train:]]
        else:
            x_train = [x, dx_dt, u, mu]
            x_val = None
        train_hist = aphin.fit(
            x=x_train,
            validation_data=x_val,
            epochs=db_cfg["n_epochs"],
            batch_size=db_cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        save_training_times(train_hist, result_dir)
        aphin_vis.plot_train_history(
            train_hist,
            save_path=result_dir,
            validation=validation,
        )

        # load best weights
        aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))

    # write data to results directory
    save_weights(weight_dir, result_dir, load_network=db_cfg["load_network"])
    write_to_experiment_overview(
        db_cfg, result_dir, load_network=db_cfg["load_network"]
    )

    # %% debugging

    # %% Validation
    logging.info(
        "################################   3. Validation ################################"
    )
    t_test, x_test, dx_dt_test, u_test, mu_test = disc_brake_data.test_data
    n_sim_test, n_t_test, _, _, _, _ = disc_brake_data.shape_test
    print_matrices(system_layer, mu=mu_test, n_t=n_t_test)

    # calculate projection and Jacobian errors
    file_name = "projection_error.txt"
    projection_error_file_dir = os.path.join(result_dir, file_name)
    aphin.get_projection_properties(x, x_test, file_dir=projection_error_file_dir)

    # %% Validation of the AE reconstruction
    # get original quantities
    disc_brake_data_id = PHIdentifiedDataset.from_identification(
        disc_brake_data,
        system_layer,
        aphin,
        integrator_type="imr",
        calc_u_midpoints=True,
    )
    save_evaluation_times(disc_brake_data_id, result_dir)

    # # %% 3D plots
    create_3d_vis = True
    if create_3d_vis:
        # save each test parameter set as csv
        for i, mu_ in enumerate(disc_brake_data.TEST.Mu):
            # save header without leading # to avoid problems with np.loadtxt
            np.savetxt(
                os.path.join(result_dir, f"test_parameter_set_{i}.csv"),
                np.array(
                    [
                        np.arange(n_mu + n_u),
                        np.concatenate([mu_, disc_brake_data.TEST.U[i, 0]]),
                    ]
                ).T,
                delimiter=",",
                header="idx,mu",
                comments="",
            )

        faces_path = os.path.join(data_dir, db_cfg["faces"])
        ref_coords_path = os.path.join(data_dir, db_cfg["ref_coords"])
        faces = np.load(faces_path)
        ref_coords = np.load(ref_coords_path)
        vis = Visualizer(background_color=(1, 1, 1, 0))

        disc_brake_data.rescale_X()
        disc_brake_data_id.rescale_X()
        # linear expansion coefficient 1e-6 to large - correct physical values manually
        disc_brake_data.scale_X(
            scaling_values=[1, 1e6, 1e6], domain_split_vals=db_cfg["domain_split_vals"]
        )
        disc_brake_data_id.scale_X(
            scaling_values=[1, 1e6, 1e6], domain_split_vals=db_cfg["domain_split_vals"]
        )

        disps_ref = disc_brake_data.TEST.X[:, :, :, 1:4]
        disps_pred = disc_brake_data_id.TEST.X[:, :, :, 1:4]
        e_disp = np.linalg.norm(disps_ref - disps_pred, axis=-1)
        vels_ref = np.linalg.norm(disc_brake_data.TEST.X[:, :, :, 4:], axis=-1)
        vels_pred = np.linalg.norm(disc_brake_data_id.TEST.X[:, :, :, 4:], axis=-1)
        e_vel = np.abs(vels_ref - vels_pred)
        temp_ref = disc_brake_data.TEST.X[:, :, :, 0]
        temp_pred = disc_brake_data_id.TEST.X[:, :, :, 0]
        e_temp = np.abs(temp_ref - temp_pred)

        # anim settings
        camera_distance = 1
        view = [80, -60]
        temp_max = 2000  # np.max(disc_brake_data.TEST.X[:, :, :, 0])
        video_dir = os.path.join(result_dir, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        ampl = 20
        time_ids = np.arange(n_t_test - 1, n_t_test, 1)

        e_temp_max = e_temp.max()  # e_temp.mean() + 3 * e_temp.std()
        e_disp_max = e_disp.max()  # e_disp.mean() + 3 * e_disp.std()
        e_vel_max = e_vel.max()  # e_vel.mean() + 3 * e_vel.std()

        videos = dict(
            ref=dict(
                disps=disps_ref,
                color=temp_ref,
                color_scale_limits=[0, temp_max],
                colormap="plasma",
            ),
            pred=dict(
                disps=disps_pred,
                color=temp_pred,
                color_scale_limits=[0, temp_max],
                colormap="plasma",
            ),
            e_disp=dict(
                disps=disps_pred,
                color=e_disp,
                color_scale_limits=[0, e_disp_max],
                colormap="viridis",
            ),
            e_vel=dict(
                disps=disps_pred,
                color=e_vel,
                color_scale_limits=[0, e_vel_max],
                colormap="viridis",
            ),
            e_temp=dict(
                disps=disps_pred,
                color=e_temp,
                color_scale_limits=[0, e_temp_max],
                colormap="viridis",
            ),
        )

        # save colorbar limits
        np.savetxt(
            os.path.join(result_dir, f"colorbar_limits.csv"),
            np.array(
                [
                    e_temp_max,
                    e_disp_max,
                    e_vel_max,
                ]
            ),
            delimiter=",",
            header="temp,disp,vel",
            comments="",
        )

        for sim_id in range(n_sim_test):
            for key, video_setting in videos.items():
                print(f"{sim_id}_{key}")
                vis.animate(
                    ampl * video_setting["disps"][sim_id, time_ids] + ref_coords[:, :],
                    faces=faces,
                    color=video_setting["color"][sim_id, time_ids],
                    camera_distance=camera_distance,
                    colormap=video_setting["colormap"],
                    color_scale_limits=video_setting["color_scale_limits"],
                    view=view,
                    save_single_frames=True,
                    save_animation=True,
                    animation_name=os.path.join(video_dir, f"{key}_sim_{sim_id}"),
                    close_on_end=True,
                    play_at_start=True,
                )

        # import plotly.graph_objects as go

        # fig = go.Figure()
        # for i_node in range(disc_brake_data.TEST.X.shape[2]):
        #     fig.add_trace(
        #         go.Scatter(
        #             x=disc_brake_data.TEST.t.flatten(),
        #             y=disc_brake_data.TEST.X[0, :, i_node, 0],
        #             mode="lines",
        #         )
        #     )
        # fig.show()

    # %% finer time discretization
    # if isinstance(system_layer, PHQLayer):
    #     J_ph, R_ph, B_ph, Q_ph = system_layer.get_system_matrices(
    #         disc_brake_data.TRAIN.mu, n_t=disc_brake_data.TRAIN.n_t
    #     )
    #     E_ph = None
    # elif isinstance(system_layer, PHLayer):
    #     J_ph, R_ph, B_ph = system_layer.get_system_matrices(
    #         disc_brake_data.TRAIN.mu, n_t=disc_brake_data.TRAIN.n_t
    #     )
    #     Q_ph = None
    #     E_ph = None
    # disc_brake_data_id_fine = copy.deepcopy(disc_brake_data_id)
    # x_vals_lin = np.arange(disc_brake_data_id_fine.TRAIN.t.shape[0])
    # finer_factor = 3
    # x_vals_fine = np.linspace(
    #     0,
    #     disc_brake_data_id_fine.TRAIN.t.shape[0] - 1,
    #     disc_brake_data_id_fine.TRAIN.t.shape[0] * finer_factor,
    # )
    # disc_brake_data_id_fine.TRAIN.t = np.expand_dims(
    #     np.interp(x_vals_fine, x_vals_lin, disc_brake_data_id_fine.TRAIN.t.ravel()),
    #     axis=1,
    # )
    # disc_brake_data_id_fine.TRAIN.n_t = disc_brake_data_id_fine.TRAIN.t.shape[0]
    # disc_brake_data_id_fine.TRAIN.n_f = (
    #     disc_brake_data_id_fine.TRAIN.n_n * disc_brake_data_id_fine.TRAIN.n_dn
    # )
    # disc_brake_data_id_fine.TRAIN.get_initial_conditions()
    # disc_brake_data_id_fine.TRAIN.mu = disc_brake_data.TRAIN.mu
    # U_out = scipy.interpolate.interp1d(
    #     x_vals_lin, disc_brake_data_id_fine.TRAIN.U, axis=1
    # )
    # disc_brake_data_id_fine.TRAIN.U = U_out(x_vals_fine)

    # (
    #     z_ph_s,
    #     dz_dt_ph_s,
    #     x_ph_s,
    #     dx_dt_ph_s,
    #     Z_ph,
    #     Z_dt_ph,
    #     X_ph,
    #     X_dt_ph,
    #     H_ph,
    #     solving_times,
    # ) = disc_brake_data_id_fine.TRAIN.obtain_ph_data(
    #     disc_brake_data_id_fine.TRAIN,
    #     aphin,
    #     system_layer,
    #     J_ph,
    #     R_ph,
    #     B_ph,
    #     Q_ph,
    #     E_ph,
    #     integrator_type="imr",
    #     decomp_option="lu",
    #     calc_u_midpoints=True,
    # )

    # Z_ph_old_interp_fine = scipy.interpolate.interp1d(
    #     x_vals_lin, disc_brake_data_id.TRAIN.Z_ph, axis=1
    # )(x_vals_fine)
    # Z_old_interp_fine = scipy.interpolate.interp1d(
    #     x_vals_lin, disc_brake_data_id.TRAIN.Z, axis=1
    # )(x_vals_fine)
    # max_diff = np.max(np.abs(Z_ph_old_interp_fine - Z_ph))

    # state_n_sim = 1
    # state_nf = 5
    # plt.figure()
    # plt.plot(
    #     disc_brake_data_id_fine.TRAIN.t,
    #     Z_ph_old_interp_fine[state_n_sim, :, state_nf],
    #     label="Z_ph",
    # )
    # plt.plot(
    #     disc_brake_data_id_fine.TRAIN.t,
    #     Z_old_interp_fine[state_n_sim, :, state_nf],
    #     label="Z",
    # )
    # plt.plot(
    #     disc_brake_data_id_fine.TRAIN.t,
    #     Z_ph[state_n_sim, :, state_nf],
    #     label="Z_ph_fine",
    # )
    # plt.legend()
    # plt.show(block=True)

    # rescale data
    # disc_brake_data.rescale_X()
    # disc_brake_data_id.rescale_X()

    use_train_data = False
    # %% calculate errors
    disc_brake_data.calculate_errors(
        disc_brake_data_id,
        domain_split_vals=db_cfg["domain_split_vals"],
        save_to_txt=True,
        result_dir=result_dir,
    )
    aphin_vis.plot_errors(
        disc_brake_data,
        use_train_data=use_train_data,
        save_name=os.path.join(result_dir, "rms_error"),
        domain_names=db_cfg["domain_names"],
        save_to_csv=True,
        yscale="log",
        only_save=db_cfg["only_save"],
    )

    # aphin_vis.single_parameter_space_error_plot(
    #     disc_brake_data.TEST.state_error_list[0],
    #     disc_brake_data.TEST.Mu,
    #     disc_brake_data.TEST.Mu_input,
    #     parameter_names=["conductivity", "density", "heat flux"],
    #     save_name="",
    # )

    # aphin_vis.single_parameter_space_error_plot(
    #     disc_brake_data.TRAIN.state_error_list[0],
    #     disc_brake_data.TRAIN.Mu,
    #     disc_brake_data.TRAIN.Mu_input,
    #     parameter_names=["conductivity", "density", "heat flux"],
    #     save_name="",
    # )

    # %% plot trajectories
    idx_gen = "rand"
    aphin_vis.plot_time_trajectories_all(
        disc_brake_data,
        disc_brake_data_id,
        use_train_data=use_train_data,
        idx_gen=idx_gen,
        result_dir=result_dir,
        only_save=db_cfg["only_save"],
    )

    create_costum_plot = False
    if create_costum_plot:
        # rescale data
        disc_brake_data.rescale_X()
        disc_brake_data_id.rescale_X()
        # linear expansion coefficient 1e-6 to large - correct physical values manually
        disc_brake_data.scale_X(
            scaling_values=[1, 1e6, 1e6], domain_split_vals=db_cfg["domain_split_vals"]
        )
        disc_brake_data_id.scale_X(
            scaling_values=[1, 1e6, 1e6], domain_split_vals=db_cfg["domain_split_vals"]
        )

        rng = np.random.default_rng(seed=db_cfg["seed"])
        n_n_costum = 2082  # between heat node
        n_n_heat_line_nodes = (
            np.array([759, 1695, 1934, 2199, 2233]) - 1
        )  # -1 for 0-indexing (node numbers from Abaqus)
        n_sim_constant = rng.integers(disc_brake_data.TEST.n_sim)
        n_sim_random = rng.integers(0, disc_brake_data.TEST.n_sim, (5,))
        index_list_sim_temp = [
            (n_sim_costum, n_n_costum, 0) for n_sim_costum in n_sim_random
        ]
        index_list_sim_dispz = [
            (n_sim_costum, n_n_costum, 3) for n_sim_costum in n_sim_random
        ]
        index_list_sim = index_list_sim_temp + index_list_sim_dispz
        index_list_nodes_temp = [
            (n_sim_constant, n_n_heat, 0) for n_n_heat in n_n_heat_line_nodes
        ]
        index_list_nodes_dispz = [
            (n_sim_constant, n_n_heat, 3) for n_n_heat in n_n_heat_line_nodes
        ]
        index_list_nodes_velz = [
            (n_sim_constant, n_n_heat, 6) for n_n_heat in n_n_heat_line_nodes
        ]
        index_list_nodes = (
            index_list_nodes_temp + index_list_nodes_dispz + index_list_nodes_velz
        )

        subplot_idx = (
            [0] * len(index_list_nodes_temp)
            + [1] * len(index_list_nodes_dispz)
            + [2] * len(index_list_nodes_dispz)
        )

        subplot_title = ["temp", "disp-z", "vel-z"]

        aphin_vis.custom_state_plot(
            data=disc_brake_data,
            data_id=disc_brake_data_id,
            attributes=["X", "X"],
            index_list=index_list_nodes,
            use_train_data="test",
            result_dir=result_dir,
            subplot_idx=subplot_idx,
            subplot_title=subplot_title,
            save_to_csv=True,
            save_name="db_custom_nodes",
        )

    # node = 1432
    # node = np.random.randint(0, 2250)
    # plt.figure()
    # plt.plot(disc_brake_data.TRAIN.X[1, :, node, :])
    # plt.plot(disc_brake_data_id.TRAIN.X_rec[1, :, node, :], "-.")
    # legend = [f"dof{i}" for i in range(7)]
    # legend.extend([f"rec{i}" for i in range(7)])
    # plt.legend()
    # plt.show(block=True)

    # avoid that the script stops and keep the plots open
    # plt.show()

    print("debug stop")
    # import plotly.graph_objects as go
    # import plotly.express as px

    # show_latent_state_num = 1
    # # Jonas dz test
    # # %% Chain rule erivative vs numerical derivative
    # z = aphin.encode(disc_brake_data.TEST.x).numpy()
    # # calc time derivatives
    # t_conc = []
    # t_start_add_value = 0
    # for i_sim in range(disc_brake_data.TEST.n_sim):
    #     dt = disc_brake_data.TEST.t[1] - disc_brake_data.TEST.t[0]
    #     t = disc_brake_data.TEST.t.squeeze(axis=1) + t_start_add_value
    #     t_start_add_value = t[-1] + dt
    #     t_conc.append(t)
    # t_conc_array = np.array(t_conc).flatten()
    # dzdt_numeric = np.gradient(z, t_conc_array, axis=0)
    # cut_idx = 1000
    # z, dzdt_chain = aphin.calc_latent_time_derivatives(
    #     disc_brake_data.TEST.x[:cut_idx], disc_brake_data.TEST.dx_dt[:cut_idx]
    # )
    # dzdt_numeric_cut = dzdt_numeric[:cut_idx]
    # i_state = 0
    # plt.figure()
    # plt.plot(dzdt_numeric_cut[:, i_state])
    # plt.plot(dzdt_chain[:, i_state], "--")
    # plt.xlabel("time")
    # plt.ylabel(r"$\dot{z}$")
    # plt.legend(["dzdt numerically", "dzdt chain rule"])
    # plt.show()
    # plt.savefig("test_z_dt_numeric_vs_chain.png")


# parameter variation for multiple experiment runs
# requires calc_various_experiments = True
def create_variation_of_parameters():
    parameter_variation_dict = {
        # "lr": [0.0000025, 0.00000025],
        # "l_dz": [10, 0.1],
        # "l_dx": [0, 0.1, 0.000001],
        # "l_rec": [10, 0.1],
        "r": [16, 2, 4, 8, 12, 24],
        # "ph_layer": ["ph", "phq"],
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
        # use standard config file - single run
        main()
