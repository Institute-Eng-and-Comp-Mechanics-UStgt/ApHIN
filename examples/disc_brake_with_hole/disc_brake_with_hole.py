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
    db_cfg = configuration.cfg_dict
    data_dir, log_dir, weight_dir, result_dir = configuration.directories

    if db_cfg["save_config_sweep_data"]:
        save_config_sweep_data(
            root_result_dir=os.path.dirname(result_dir),
            common_folder_name=db_cfg["common_folder_name"],
            sweep_key=db_cfg["r"],
            domain_names=db_cfg["domain_names"],
        )

    aphin_vis.setup_matplotlib(db_cfg["setup_matplotlib"])
    # %% Data
    logging.info(
        "################################   1. Data ################################"
    )
    # save/load data path
    # sim_name = db_cfg["sim_name"]
    data_name = db_cfg["data_name"]
    if not ("data_dir" not in db_cfg or db_cfg["data_dir"] is None):
        data_dir = db_cfg["data_dir"]
    cache_path = os.path.join(data_dir, f"{data_name}")  # path to .npz file

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
    test_size = db_cfg["test_size"]
    disc_brake_data.train_test_split(test_size=test_size, seed=db_cfg["seed"])

    # decrease number of simulations
    if db_cfg["num_sim"] is not None:
        disc_brake_data.decrease_num_simulations(
            num_sim=db_cfg["num_sim"], seed=db_cfg["seed"]
        )

    if db_cfg["num_time_steps"] is not None:
        logging.info("Number of time steps is reduced.")
        disc_brake_data.decrease_num_time_steps(num_time_steps=db_cfg["num_time_steps"])

    if db_cfg["cut_time_start_and_end"]:
        disc_brake_data.cut_time_start_and_end()

    # scale data
    disc_brake_data.scale_X(
        scaling_values=db_cfg["scaling_values"],
        domain_split_vals=db_cfg["domain_split_vals"],
    )
    if not db_cfg["sim_name"] == "test_data":
        disc_brake_data.scale_Mu(
            mu_train_bounds=None, desired_bounds=db_cfg["desired_bounds"]
        )

    disc_brake_data.scale_U_domain_wise()

    # transform to feature form that is used by the deep learning
    disc_brake_data.states_to_features()
    t, x, dx_dt, u, mu = disc_brake_data.data

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
    if db_cfg["create_3d_vis"]:
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

    if db_cfg["create_costum_plot"]:
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
        # n_n_costum = 2082  # between heat node
        n_n_heat_line_nodes = (
            np.array([759, 1695, 1934, 2199, 2233]) - 1
        )  # -1 for 0-indexing (node numbers from Abaqus)
        n_sim_constant = rng.integers(disc_brake_data.TEST.n_sim)
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

    # avoid that the script stops and keep the plots open
    plt.show()


# parameter variation for multiple experiment runs
# requires calc_various_experiments = True
def create_variation_of_parameters():
    parameter_variation_dict = {
        "r": [2, 4, 8, 12, 16, 24],
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
