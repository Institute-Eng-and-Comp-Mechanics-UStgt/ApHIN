# default packages
import numpy as np
import logging
import os

# third party packages
import tensorflow as tf

# own packages
from visualizer import Visualizer
from aphin.utils.data import PHIdentifiedDataset, DiscBrakeDataset
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
)
from aphin.identification import APHIN
from aphin.layers.phq_layer import PHQLayer
from aphin.utils.configuration import Configuration
import aphin.utils.visualizations as aphin_vis
from aphin.utils.experiments import run_various_experiments
from aphin.utils.callbacks_tensorflow import callbacks
from aphin.utils.print_matrices import print_matrices
import urllib.request


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

    aphin_vis.setup_matplotlib(db_cfg["setup_matplotlib"])
    # %% Data
    logging.info(
        "################################   1. Data ################################"
    )
    # save/load data path
    sim_name = db_cfg["sim_name"]
    cache_path = os.path.join(data_dir, f"{sim_name}.npz")  # path to .npz file

    # check if data already exists on local machine
    if not os.path.isfile(cache_path):
        # download file if it is missing
        file_url = db_cfg["file_url"]
        logging.info(f"download data from {file_url} and save it to {cache_path}.")
        urllib.request.urlretrieve(
            file_url,
            cache_path,
        )

    # reduced size
    r = db_cfg["r"]

    # %% load/create data
    # Initialize Dataset

    create_data_from_txt = db_cfg[
        "create_data_from_txt"
    ]  # True if data has not been preprocessed to an .npz file before
    if create_data_from_txt:
        # create data from .txt files
        # idx of parameters in parameter file
        idx_mu = np.arange(db_cfg["n_mu"])
        disc_brake_txt_path = config.disc_brake_txt_path
        disc_brake_txt_path = os.path.join(disc_brake_txt_path, sim_name)
        disc_brake_data = DiscBrakeDataset.from_txt(
            disc_brake_txt_path,
            save_cache=True,
            cache_path=cache_path,
            idx_mu=idx_mu,
            use_velocities=db_cfg["use_velocities"],
        )
    else:
        # load .npz file
        disc_brake_data = DiscBrakeDataset.from_data(
            cache_path, use_velocities=db_cfg["use_velocities"]
        )

    # %%
    # split into train and test data
    train_test_split_method = db_cfg["train_test_split_method"]
    if train_test_split_method == "sim_idx":
        sim_idx_train = np.arange(50)
        sim_idx_test = np.arange(10) + len(sim_idx_train)
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

    # filter data with savgol filter
    if db_cfg["filter_data"]:
        logging.info("Data is filtered")
        disc_brake_data.filter_data(interp_equidis_t=False)
    else:
        logging.info("Data is not filtered.")

    if db_cfg["num_time_steps"] is not None:
        logging.info("Number of time steps is reduced.")
        disc_brake_data.decrease_num_time_steps(num_time_steps=db_cfg["num_time_steps"])

    # scale data
    disc_brake_data.scale_all(
        scaling_values=db_cfg["scaling_values"],
        domain_split_vals=db_cfg["domain_split_vals"],
        u_desired_bounds=db_cfg["desired_bounds"],
        mu_desired_bounds=db_cfg["desired_bounds"],
    )
    # transform to feature form that is used by the deep learning
    disc_brake_data.states_to_features()
    t, x, dx_dt, u, mu = disc_brake_data.data

    # %% Create APHIN
    logging.info(
        "################################   2. Model      ################################"
    )

    callback = callbacks(
        weight_dir,
        tensorboard=db_cfg["tensorboard"],
        log_dir=log_dir,
        monitor="loss",
        earlystopping=True,
        patience=500,
    )

    n_sim, n_t, n_n, n_dn, n_u, n_mu = disc_brake_data.shape
    regularizer = tf.keras.regularizers.L1L2(l1=db_cfg["l1"], l2=db_cfg["l2"])
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

    # Fit or learn neural network
    if db_cfg["load_network"]:
        logging.info(f"Loading NN weights.")
        aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    else:
        logging.info(f"Fitting NN weights.")
        train_hist = aphin.fit(
            x=[x, dx_dt, u, mu],
            # validation_data=([x, u_x, dx_dt], None),
            epochs=db_cfg["n_epochs"],
            batch_size=db_cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        aphin_vis.plot_train_history(train_hist)

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
        disc_brake_data, system_layer, aphin, integrator_type="imr"
    )

    # %% calculate errors
    disc_brake_data.calculate_errors(
        disc_brake_data_id,
        domain_split_vals=db_cfg["domain_split_vals"],
        save_to_txt=True,
        result_dir=result_dir,
    )
    aphin_vis.plot_errors(
        disc_brake_data,
        t=disc_brake_data.t_test,
        save_name=os.path.join(result_dir, "rms_error"),
        domain_names=db_cfg["domain_names"],
        save_to_csv=True,
        yscale="log",
    )

    # %% 3D plots
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

    # rescale data
    disc_brake_data.rescale_X()
    disc_brake_data_id.is_scaled = True
    disc_brake_data_id.TRAIN.is_scaled = True
    disc_brake_data_id.TEST.is_scaled = True
    disc_brake_data_id.scaling_values = disc_brake_data.TRAIN.scaling_values
    disc_brake_data_id.TRAIN.scaling_values = disc_brake_data.TRAIN.scaling_values
    disc_brake_data_id.TEST.scaling_values = disc_brake_data.TRAIN.scaling_values
    disc_brake_data_id.rescale_X()

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
    camera_distance = 0.2
    view = [45, 0]
    temp_max = 1500
    video_dir = os.path.join(result_dir, "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    ampl = 400
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
            print(key)
            vis.animate(
                ampl * disps_ref[sim_id, time_ids] + ref_coords[:, 1:],
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
            )

    # %% plot trajectories
    use_train_data = False
    idx_gen = "rand"
    aphin_vis.plot_time_trajectories_all(
        disc_brake_data,
        disc_brake_data_id,
        use_train_data=use_train_data,
        idx_gen=idx_gen,
        result_dir=result_dir,
    )


# parameter variation for multiple experiment runs
# requires calc_various_experiments = True
def create_variation_of_parameters():
    parameter_variation_dict = {
        "l_rec": [1, 0.1, 10],
        "l_dz": [0.1, 0.01, 1],
        "l_dx": [0.001, 0.0001, 0.01],
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
