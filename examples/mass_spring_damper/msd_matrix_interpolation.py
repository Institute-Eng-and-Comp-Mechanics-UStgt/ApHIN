# default packages
import logging
import os
import numpy as np
import copy
from scipy.spatial import ConvexHull


# third party packages
import tensorflow as tf

# import matplotlib
# matplotlib.use("TkAgg")  # Force interactive backend
import matplotlib.pyplot as plt

# own packages
import aphin.utils.visualizations as aphin_vis
from aphin.identification import PHIN
from aphin.layers import PHLayer, PHQLayer, LTILayer
from aphin.utils.data import Dataset, PHIdentifiedDataset, PHIdentifiedData
from aphin.utils.callbacks_tensorflow import callbacks
from aphin.utils.configuration import Configuration
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
    save_evaluation_times,
    save_training_times,
)
from aphin.utils.transformations import (
    reshape_states_to_features,
    reshape_features_to_states,
)
from aphin.utils.print_matrices import print_matrices
from data_generation.matrix_interpolation import (
    get_coeff_values,
    evaluate_interpolation,
    get_weighting_function_values,
    evaluate_matrices,
    in_hull,
)
from aphin.systems import LTISystem, PHSystem
from aphin.utils.experiments import run_various_experiments


# tf.config.run_functions_eagerly(True)


def phin_learning(
    msd_data, msd_cfg, config_dirs, dir_extension: str = "", layer: str = "phq_layer"
):

    data_dir, log_dir, weight_dir, result_dir = config_dirs
    log_dir = os.path.join(log_dir, dir_extension)
    weight_dir = os.path.join(weight_dir, dir_extension)
    result_dir = os.path.join(result_dir, dir_extension)
    for dir in [log_dir, weight_dir, result_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    t, x, dx_dt, u, mu = msd_data.data
    n_sim, n_t, n_n, n_dn, n_u, n_mu = msd_data.shape
    n_f = n_n * n_dn

    # %% Create PHAutoencoder
    logging.info(
        "################################   2. Model      ################################"
    )

    validation = msd_cfg["validation"]
    if validation:
        monitor = "val_loss"
    else:
        monitor = "loss"

    # ph identification network (pHIN)
    callback = callbacks(
        weight_dir,
        tensorboard=msd_cfg["tensorboard"],
        log_dir=log_dir,
        monitor=monitor,
        earlystopping=True,
        patience=500,
    )

    # %% Create PHAutoencoder
    logging.info(
        "################################   2. Model      ################################"
    )

    regularizer = tf.keras.regularizers.L1L2(l1=msd_cfg["l1"], l2=msd_cfg["l2"])
    if layer == "phq_layer":
        system_layer = PHQLayer(
            n_f,
            n_u=n_u,
            n_mu=n_mu,
            regularizer=regularizer,
            name="phq_layer",
            layer_sizes=msd_cfg["layer_sizes_ph"],
            activation=msd_cfg["activation_ph"],
        )
    elif layer == "ph_layer":
        system_layer = PHLayer(
            n_f,
            n_u=n_u,
            n_mu=n_mu,
            regularizer=regularizer,
            name="ph_layer",
            layer_sizes=msd_cfg["layer_sizes_ph"],
            activation=msd_cfg["activation_ph"],
        )
    elif layer == "lti_layer":
        system_layer = LTILayer(
            n_f,
            n_u=n_u,
            n_mu=n_mu,
            regularizer=regularizer,
            name="lti_layer",
            layer_sizes=msd_cfg["layer_sizes_ph"],
            activation=msd_cfg["activation_ph"],
        )
    else:
        raise ValueError(f"Unknown layer {layer}.")

    phin = PHIN(n_f, x=x, u=u, mu=mu, system_layer=system_layer, name="phin")

    #  create model with several inputs
    # TODO: Move back to ADAM if not working
    phin.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=msd_cfg["lr"]),
        loss=tf.keras.losses.MSE,
    )
    if mu is None:
        input_shape = [x.shape, dx_dt.shape, u.shape]
    else:
        input_shape = [x.shape, dx_dt.shape, u.shape, mu.shape]
    phin.build(input_shape=(input_shape, None))

    # phin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    # phin.load_weights(data_path_weights_filename)
    if msd_cfg["load_network"]:
        logging.info(f"Loading NN weights.")
        phin.load_weights(os.path.join(weight_dir, ".weights.h5"))
    else:
        if validation:
            n_train = int(0.8 * x.shape[0])
            if mu is None:
                x_train = [x[:n_train], dx_dt[:n_train], u[:n_train]]
                x_val = ([x[n_train:], dx_dt[n_train:], u[n_train:]], None)
            else:
                x_train = [x[:n_train], dx_dt[:n_train], u[:n_train], mu[:n_train]]
                x_val = (
                    [x[n_train:], dx_dt[n_train:], u[n_train:], mu[n_train:]],
                    None,
                )
        else:
            if mu is None:
                x_train = [x, dx_dt, u]
            else:
                x_train = [x, dx_dt, u, mu]
            x_val = None
        logging.info(f"Fitting NN weights.")
        train_hist = phin.fit(
            x=x_train,
            validation_data=x_val,
            epochs=msd_cfg["n_epochs"],
            batch_size=msd_cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        save_training_times(train_hist, result_dir)
        aphin_vis.plot_train_history(
            train_hist, save_path=result_dir, validation=msd_cfg["validation"]
        )
        phin.load_weights(os.path.join(weight_dir, ".weights.h5"))

    # write data to results directory
    save_weights(weight_dir, result_dir, load_network=msd_cfg["load_network"])
    write_to_experiment_overview(
        msd_cfg, result_dir, load_network=msd_cfg["load_network"]
    )

    # calculate all needed identified results
    msd_data_id = PHIdentifiedDataset.from_identification(msd_data, system_layer, phin)
    return msd_data_id, system_layer, phin


def main(config_path_to_file=None, only_phin: bool = False):
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
    if manual_results_folder is not None:
        # changed priority
        config_info = manual_results_folder
    elif config_path_to_file is not None:
        config_info = config_path_to_file
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
            f"File could not be found. If this is the first time you run this example, please execute the data generating script `./data_generation/mass_spring_damper_data_gen.py` first."
        )

    msd_data.train_test_split_convex_hull(
        desired_min_num_train=60,
        n_simulations_per_parameter_set=msd_cfg["n_simulations_per_parameter_set"],
    )

    # scale data
    msd_data.scale_Mu(desired_bounds=msd_cfg["desired_bounds"])
    if msd_cfg["scale_x"]:
        msd_data.scale_X(domain_split_vals=msd_cfg["domain_split_vals"])
        msd_data.scale_U(desired_bounds=[-1, 1])
    msd_data.states_to_features()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%                        PHIN                                            %%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # usual PHIN framework with mu DNN
    dir_extension = "phin"
    msd_data_id_phin, system_layer_phin, phin_phin = phin_learning(
        msd_data,
        msd_cfg,
        configuration.directories,
        dir_extension=dir_extension,
        layer=msd_cfg["ph_layer"],
    )

    result_dir_phin = os.path.join(result_dir, dir_extension)
    if not os.path.isdir(result_dir_phin):
        os.mkdir(result_dir_phin)
    use_train_data = False
    idx_gen = "rand"
    msd_data.calculate_errors(
        msd_data_id_phin,
        save_to_txt=True,
        result_dir=result_dir_phin,
        domain_split_vals=[1, 1],
    )
    aphin_vis.plot_time_trajectories_all(
        msd_data, msd_data_id_phin, use_train_data, idx_gen, result_dir_phin
    )
    aphin_vis.plot_errors(
        msd_data,
        use_train_data,
        save_name=os.path.join(result_dir_phin, "rms_error"),
        save_to_csv=True,
    )
    aphin_vis.single_parameter_space_error_plot(
        msd_data.TEST.state_error_list[0],
        msd_data.TEST.Mu,
        msd_data.TEST.Mu_input,
        parameter_names=["mass", "stiff", "omega", "delta"],
        save_path=result_dir_phin,
    )

    msd_data_id_phin.TEST.add_system_matrices_from_system_layer(
        system_layer=system_layer_phin
    )

    eig_vals_phin = msd_data_id_phin.TEST.calculate_eigenvalues(
        result_dir=result_dir_phin,
        save_to_csv=True,
        save_name="eigenvalues_phin",
    )
    # permute matrices to fit the publication
    perm = [0, 3, 1, 4, 2, 5]
    msd_data.permute_matrices(permutation_idx=perm)
    # test_ids = [0, 1, 3, 6, 7]  # test_ids = range(10) # range(6) test_ids = [0]
    rng = np.random.default_rng(seed=msd_cfg["seed"])
    test_ids = rng.integers(0, msd_data.TEST.n_sim, size=(5,))
    # phin
    aphin_vis.chessboard_visualisation(
        test_ids,
        msd_data,
        system_layer=system_layer_phin,
        result_dir=result_dir_phin,
        limits=msd_cfg["matrix_color_limits"],
        error_limits=msd_cfg["matrix_error_limits"],
    )

    create_costum_plot = True
    if create_costum_plot:
        rng = np.random.default_rng()
        n_n_array = np.arange(3)
        # n_sim_constant = rng.integers(msd_data.TEST.n_sim)
        n_dn = 0  # displacements
        n_sim_random = rng.integers(0, msd_data.TEST.n_sim, (5,))

        index_list_disps = []
        subplot_idx = []
        for i_sim, n_sim in enumerate(n_sim_random):
            for n_n in n_n_array:
                subplot_idx.append(i_sim)
                index_list_disps.append((n_sim, n_n, n_dn))

        subplot_title = [f"n_sim{sim_num}" for sim_num in n_sim_random]

        aphin_vis.custom_state_plot(
            data=msd_data,
            data_id=msd_data_id_phin,
            attributes=["X", "X"],
            index_list=index_list_disps,
            use_train_data="test",
            result_dir=result_dir_phin,
            subplot_idx=subplot_idx,
            subplot_title=subplot_title,
            save_to_csv=True,
            save_name="msd_custom_nodes_phin",
        )

    if only_phin:
        return

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%                        LTI                                             %%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dir_extension = "lti"
    msd_data_id_lti, system_layer_lti, phin_lti = phin_learning(
        msd_data,
        msd_cfg,
        configuration.directories,
        dir_extension=dir_extension,
        layer="lti_layer",
    )

    result_dir_lti = os.path.join(result_dir, dir_extension)
    if not os.path.isdir(result_dir_lti):
        os.mkdir(result_dir_lti)
    use_train_data = False
    idx_gen = "rand"
    msd_data.calculate_errors(
        msd_data_id_lti,
        save_to_txt=True,
        result_dir=result_dir_lti,
        domain_split_vals=[1, 1],
    )
    aphin_vis.plot_time_trajectories_all(
        msd_data, msd_data_id_lti, use_train_data, idx_gen, result_dir_lti
    )
    aphin_vis.plot_errors(
        msd_data,
        use_train_data,
        save_name=os.path.join(result_dir_lti, "rms_error"),
        save_to_csv=True,
    )
    aphin_vis.single_parameter_space_error_plot(
        msd_data.TEST.state_error_list[0],
        msd_data.TEST.Mu,
        msd_data.TEST.Mu_input,
        parameter_names=["mass", "stiff", "omega", "delta"],
        save_path=result_dir_lti,
    )

    msd_data_id_lti.TEST.add_system_matrices_from_system_layer(
        system_layer=system_layer_lti
    )

    eig_vals_lti = msd_data_id_lti.TEST.calculate_eigenvalues(
        result_dir=result_dir_lti,
        save_to_csv=True,
        save_name="eigenvalues_lti",
    )

    aphin_vis.chessboard_visualisation(
        test_ids,
        msd_data,
        system_layer=system_layer_lti,
        result_dir=result_dir_lti,
        limits=msd_cfg["matrix_color_limits"],
        error_limits=msd_cfg["matrix_error_limits"],
    )

    # # 3d plot of initial conditions
    # fig = plt.figure()
    # x0 = msd_data.TRAIN.X[:, 0, :, 0]
    # x0_test = msd_data.TEST.X[:, 0, :, 0]
    # plt.scatter(x0[:, 0], x0[:, 1])
    # plt.scatter(x0_test[:, 0], x0_test[:, 1])
    # plt.xlabel("position mass 1")
    # plt.ylabel("position mass 2")
    # # plt.view([90,0])
    # # ax.set_zlim(-0.002, 0.002)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("initial_conditions")

    # # plot of parameters
    # fig = plt.figure()
    # plt.plot(msd_data.TRAIN.Mu[:, 0], msd_data.TRAIN.Mu[:, 1], "o")
    # plt.plot(msd_data.TEST.Mu[:, 0], msd_data.TEST.Mu[:, 1], "o")
    # plt.xlabel("k")
    # plt.ylabel("c")
    # plt.tight_layout()
    # plt.show()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%                        Matrix interpolation (MI)                       %%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    load_matrices = False
    if load_matrices:
        matrices = np.load(os.path.join(result_dir, "JRBQ_matrix_interpolation.npz"))
        J_all = matrices["J"]
        R_all = matrices["R"]
        Q_all = matrices["Q"]
        B_all = matrices["B"]
        A_all = matrices["A"]
        parameter_array = matrices["parameter_array"]

    else:
        # fit matrices
        num_train_scenarios_with_same_mu = (
            len(msd_data.TRAIN.sim_idx) // msd_cfg["n_simulations_per_parameter_set"]
        )
        msd_data_orig = copy.deepcopy(msd_data)

        parameter_array = np.zeros(
            (num_train_scenarios_with_same_mu, msd_data_orig.TRAIN.n_mu)
        )
        A_all = np.zeros(
            (
                num_train_scenarios_with_same_mu,
                msd_data.TRAIN.n_f,
                msd_data.TRAIN.n_f,
            )
        )
        Q_all = np.zeros(
            (
                num_train_scenarios_with_same_mu,
                msd_data.TRAIN.n_f,
                msd_data.TRAIN.n_f,
            )
        )
        J_all = np.zeros(
            (
                num_train_scenarios_with_same_mu,
                msd_data.TRAIN.n_f,
                msd_data.TRAIN.n_f,
            )
        )
        R_all = np.zeros(
            (
                num_train_scenarios_with_same_mu,
                msd_data.TRAIN.n_f,
                msd_data.TRAIN.n_f,
            )
        )
        B_all = np.zeros(
            (
                num_train_scenarios_with_same_mu,
                msd_data.TRAIN.n_f,
                msd_data.TRAIN.n_u,
            )
        )
        for i_same_mu_scenario in range(num_train_scenarios_with_same_mu):

            msd_data = copy.deepcopy(msd_data_orig)
            # there are always 3 scenarios with the same mu but different inputs
            sim_idx = (i_same_mu_scenario) * msd_cfg[
                "n_simulations_per_parameter_set"
            ] + np.arange(msd_cfg["n_simulations_per_parameter_set"])
            msd_data.decrease_num_simulations(sim_idx=sim_idx)
            mu_orig = msd_data.TRAIN.Mu[0, :]
            # constant parameter learning -> remove mu from dataset
            msd_data.remove_mu()
            msd_data_id_mi_scenario, system_layer_mi, phin_mi = phin_learning(
                msd_data,
                msd_cfg,
                configuration.directories,
                dir_extension=os.path.join(
                    "mi", "const_mu_runs", f"mi{i_same_mu_scenario}"
                ),
                layer=msd_cfg["ph_layer"],
            )
            if msd_cfg["ph_layer"] == "ph_layer":
                J, R, B = system_layer_mi.get_system_matrices(
                    None, msd_data_id_mi_scenario.TRAIN.n_t
                )
                Q = np.eye(J.shape[1])
            elif msd_cfg["ph_layer"] == "phq_layer":
                J, R, B, Q = system_layer_mi.get_system_matrices(
                    None, msd_data_id_mi_scenario.TRAIN.n_t
                )
            J_all[i_same_mu_scenario, :, :] = J
            R_all[i_same_mu_scenario, :, :] = R
            B_all[i_same_mu_scenario, :, :] = B
            Q_all[i_same_mu_scenario, :, :] = Q
            A_all[i_same_mu_scenario, :, :] = (J - R) @ Q
            parameter_array[i_same_mu_scenario, :] = mu_orig

        save_matrices = False
        if save_matrices:
            np.savez(
                os.path.join(
                    result_dir, f"JRBQ_matrix_interpolation_{msd_cfg['experiment']}.npz"
                ),
                J=J_all,
                R=R_all,
                Q=Q_all,
                B=B_all,
                A=A_all,
                parameter_array=parameter_array,
            )

    # test with sample data:
    test_with_sample_data = False
    if test_with_sample_data:
        weighting_array = get_weighting_function_values(
            parameter_array, parameter_array, ansatz=msd_cfg["ansatz"]
        )
        matrices_eval = evaluate_matrices(A_all, weighting_array)
        print(np.allclose(A_all, matrices_eval))  # should be True

    for TEST_or_TRAIN in ["TRAIN", "TEST"]:  # ["TEST", "TRAIN"]
        _, idx = np.unique(
            getattr(msd_data_orig, TEST_or_TRAIN).Mu, axis=0, return_index=True
        )
        parameter_mi = getattr(msd_data_orig, TEST_or_TRAIN).Mu[np.sort(idx), :]

        # %% Interpolate matrices
        weighting_array = get_weighting_function_values(
            parameter_array, parameter_mi, ansatz=msd_cfg["ansatz"]
        )
        # evaluate matrices
        if msd_cfg["matrix_type"] == "lti":
            A_mi = evaluate_matrices(A_all, weighting_array)
            B_mi = evaluate_matrices(B_all, weighting_array)
        elif msd_cfg["matrix_type"] == "ph":
            J_mi = evaluate_matrices(J_all, weighting_array)
            R_mi = evaluate_matrices(R_all, weighting_array)
            Q_mi = evaluate_matrices(Q_all, weighting_array)
            B_mi = evaluate_matrices(B_all, weighting_array)

        # %% Create data file
        # latent_shape = (
        #     getattr(msd_data_orig, TEST_or_TRAIN).n_sim,
        #     getattr(msd_data_orig, TEST_or_TRAIN).n_t,
        #     system_layer_phin.r,
        # )
        # Z_ph = np.zeros(latent_shape)
        # Z_dt_ph = np.zeros(latent_shape)
        system_lti_or_ph = []
        for i_test in range(weighting_array.shape[1]):
            # sim_indices = (i_test) * msd_cfg[
            # "n_simulations_per_parameter_set"
            # ] + np.arange(msd_cfg["n_simulations_per_parameter_set"])
            if msd_cfg["matrix_type"] == "lti":
                system_lti_or_ph.append(
                    LTISystem(A=A_mi[i_test, :, :], B=B_mi[i_test, :, :])
                )
            elif msd_cfg["matrix_type"] == "ph":
                if msd_cfg["ph_layer"] == "phq_layer":
                    system_lti_or_ph.append(
                        PHSystem(
                            J_ph=J_mi[i_test, :, :],
                            R_ph=R_mi[i_test, :, :],
                            Q_ph=Q_mi[i_test, :, :],
                            B=B_mi[i_test, :, :],
                        )
                    )
                if msd_cfg["ph_layer"] == "ph_layer":
                    system_lti_or_ph.append(
                        PHSystem(
                            J_ph=J_mi[i_test, :, :],
                            R_ph=R_mi[i_test, :, :],
                            B=B_mi[i_test, :, :],
                        )
                    )
        if TEST_or_TRAIN == "TRAIN":
            system_list_train = system_lti_or_ph.copy()
        else:
            system_list_test = system_lti_or_ph.copy()

    msd_data_id_mi = PHIdentifiedDataset.from_system_list(
        system_list_train=system_list_train,
        system_list_test=system_list_test,
        data=msd_data_orig,
    )
    #     for i_sim in sim_indices:
    #         u = getattr(msd_data_orig, TEST_or_TRAIN).U[i_sim]
    #         getattr(msd_data_orig, TEST_or_TRAIN).get_initial_conditions()
    #         x_init = np.expand_dims(
    #             getattr(msd_data_orig, TEST_or_TRAIN).x_init[i_sim, :], axis=0
    #         ).T
    #         Z_ph[i_sim], Z_dt_ph[i_sim] = system_lti_or_ph.solve_dt(
    #             getattr(msd_data_orig, TEST_or_TRAIN).t,
    #             x_init,
    #             u,
    #         )
    # z_ph, dz_dt_ph = reshape_states_to_features(Z_ph, Z_dt_ph)
    # x_ph, dx_dt_ph = z_ph, dz_dt_ph
    # X_ph, X_dt_ph = reshape_features_to_states(
    #     x_ph,
    #     getattr(msd_data_orig, TEST_or_TRAIN).n_sim,
    #     getattr(msd_data_orig, TEST_or_TRAIN).n_t,
    #     x_dt=dx_dt_ph,
    #     n_n=getattr(msd_data_orig, TEST_or_TRAIN).n_n,
    #     n_dn=getattr(msd_data_orig, TEST_or_TRAIN).n_dn,
    # )
    # z = getattr(msd_data_orig, TEST_or_TRAIN).x
    # Z = reshape_features_to_states(
    #     z,
    #     getattr(msd_data_orig, TEST_or_TRAIN).n_sim,
    #     getattr(msd_data_orig, TEST_or_TRAIN).n_t,
    #     n_f=system_layer_phin.r,
    # )
    # z_dt = getattr(msd_data_orig, TEST_or_TRAIN).dx_dt
    # Z_dt = reshape_features_to_states(
    #     z_dt,
    #     getattr(msd_data_orig, TEST_or_TRAIN).n_sim,
    #     getattr(msd_data_orig, TEST_or_TRAIN).n_t,
    #     n_f=system_layer_phin.r,
    # )
    # msd_data_id_mi = PHIdentifiedDataset()
    # setattr(
    #     msd_data_id_mi,
    #     TEST_or_TRAIN,
    #     PHIdentifiedData(
    #         t=getattr(msd_data_orig, TEST_or_TRAIN).t,
    #         X=X_ph,
    #         X_dt=X_dt_ph,
    #         Z=Z,
    #         Z_dt=Z_dt,
    #         Z_ph=Z_ph,
    #         Z_dt_ph=Z_dt_ph,
    #         Mu=getattr(msd_data_orig, TEST_or_TRAIN).Mu,
    #     ),
    # )
    # getattr(msd_data_id_mi, TEST_or_TRAIN).states_to_features()
    for TEST_or_TRAIN in ["TRAIN", "TEST"]:  # ["TEST", "TRAIN"]
        idx_gen = "rand"

        # %% get results
        if TEST_or_TRAIN == "TEST":
            result_dir_mi = os.path.join(result_dir, "mi", "test")
            use_train_data = False
        else:
            result_dir_mi = os.path.join(result_dir, "mi", "train")
            use_train_data = True

        if not os.path.isdir(result_dir_mi):
            os.makedirs(result_dir_mi)

        msd_data_orig.calculate_errors(
            msd_data_id_mi,
            domain_split_vals=[1, 1],
            save_to_txt=True,
            result_dir=result_dir_mi,
        )
        # getattr(msd_data_orig, TEST_or_TRAIN).calculate_errors(
        #     getattr(msd_data_id_mi, TEST_or_TRAIN),
        #     # save_to_txt=True,
        #     # result_dir=result_dir_mi,
        #     domain_split_vals=[1, 1],
        # )
        # with open(
        #     os.path.join(result_dir_mi, "mean_error_measures.txt"), "w"
        # ) as text_file:
        #     print(
        #         f"{TEST_or_TRAIN} state error mean: {getattr(msd_data_orig,TEST_or_TRAIN).state_error_mean}",
        #         file=text_file,
        #     )
        #     print(
        #         f"{TEST_or_TRAIN} latent error mean: {getattr(msd_data_orig,TEST_or_TRAIN).latent_error_mean}",
        #         file=text_file,
        #     )
        aphin_vis.plot_time_trajectories_all(
            msd_data_orig,
            msd_data_id_mi,
            use_train_data,
            idx_gen,
            result_dir_mi,
        )
        aphin_vis.plot_errors(
            msd_data_orig,
            use_train_data,
            save_name=os.path.join(result_dir_mi, "rms_error"),
            save_to_csv=True,
        )

        aphin_vis.custom_state_plot(
            data=msd_data_orig,
            data_id=msd_data_id_mi,
            attributes=["X", "X"],
            index_list=index_list_disps,
            use_train_data=use_train_data,
            result_dir=result_dir_mi,
            subplot_idx=subplot_idx,
            subplot_title=subplot_title,
            save_to_csv=True,
            save_name="msd_custom_nodes_mi",
        )
        # J_pred = np.transpose(J_mi, (2, 0, 1))
        # R_pred = np.transpose(R_mi, (2, 0, 1))
        # B_pred = np.transpose(B_mi, (2, 0, 1))
        # if msd_cfg["ph_layer"] == "ph_layer":
        #     A_pred = J_pred - R_pred
        # else:
        #     raise NotImplementedError("Add PHQ layer.")

        # getattr(msd_data_id_mi, TEST_or_TRAIN).add_ph_matrices(
        #     J=J_pred, R=R_pred, B=B_pred
        # )

        eig_vals_mi = getattr(msd_data_id_mi, TEST_or_TRAIN).calculate_eigenvalues(
            result_dir=result_dir_mi,
            save_to_csv=True,
            save_name="eigenvalues_pred_mi",
        )
        eig_vals_ref = getattr(msd_data_orig, TEST_or_TRAIN).calculate_eigenvalues(
            result_dir=result_dir, save_to_csv=True, save_name="eigenvalues_ref"
        )

        for i in test_ids:
            plt.figure()
            plt.plot(eig_vals_mi[i].real, eig_vals_mi[i].imag, "x", label="mi")
            plt.plot(eig_vals_ref[i].real, eig_vals_ref[i].imag, "o", label="ref")
            plt.plot(
                eig_vals_phin[i].real,
                eig_vals_phin[i].imag,
                "*",
                label="phin",
            )
            plt.plot(
                eig_vals_lti[i].real,
                eig_vals_lti[i].imag,
                "^",
                label="lti",
            )
            plt.xlabel("real")
            plt.ylabel("imag")
            plt.legend()
            plt.show(block=False)
            aphin_vis.save_as_png(os.path.join(result_dir, f"eigenvalues_{i}"))

        # state data
        # reference data
        for dof in range(3):
            # identified data
            getattr(msd_data_id_mi, TEST_or_TRAIN).save_state_traj_as_csv(
                result_dir_mi,
                second_oder=True,
                dof=dof,
                filename=f"state_{dof}_trajectories_mi",
            )

        if TEST_or_TRAIN == "TEST":
            J_pred, R_pred, Q_pred, B_pred = msd_data_id_mi.TEST.ph_matrices
            # only implemented for test case
            aphin_vis.chessboard_visualisation(
                test_ids,
                msd_data_orig,
                matrices_pred=(J_pred, R_pred, B_pred, Q_pred),
                result_dir=result_dir_mi,
                limits=msd_cfg["matrix_color_limits"],
                error_limits=msd_cfg["matrix_error_limits"],
            )

    # avoid that the script stops and keep the plots open
    plt.show()

    print("debug breakpoint")

    # msd_data_jonas = Dataset.from_data(
    #     "/scratch/tmp/jrettberg/Projects/ApHIN_Review/ApHIN/examples/mass_spring_damper/data/MSD_Qeye_ph_input_siso.npz"
    # )
    # msd_data_jonas.train_test_split(test_size=0.06, seed=1)
    # msd_data_jonas.states_to_features()

    # import plotly.graph_objects as go

    # trajectory = 1
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         x=np.arange(msd_data.TRAIN.x.shape[0]),
    #         y=msd_data.TRAIN.x[:, trajectory],
    #         mode="lines",
    #         name="msd_mi",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=np.arange(msd_data_jonas.TRAIN.x.shape[0]),
    #         y=msd_data_jonas.TRAIN.x[:, trajectory],
    #         mode="lines",
    #         name="msd_jonas",
    #     )
    # )
    # fig.show()


def create_variation_of_parameters():
    parameter_variation_dict = {
        "n_epochs": [1500, 6000],
        "l1": [0.0000001, 0.00001, 0.000000001],
        "ph_layer": ["ph_layer", "phq_layer"],
    }
    return parameter_variation_dict


if __name__ == "__main__":
    working_dir = os.path.dirname(__file__)
    calc_various_experiments = False
    only_phin = False
    if calc_various_experiments:
        logging.info(f"Multiple simulation runs...")
        # Run multiple simulation runs defined by parameter_variavation_dict
        configuration = Configuration(working_dir)
        _, log_dir, _, result_dir = configuration.directories

        run_various_experiments(
            experiment_main_script=main,  # main without parentheses
            parameter_variation_dict=create_variation_of_parameters(),
            basis_config_yml_path=os.path.join(
                os.path.dirname(__file__), "config_msd_matrix_interpolation.yml"
            ),
            result_dir=result_dir,
            log_dir=log_dir,
            force_calculation=False,
            only_phin=only_phin,
        )
    else:
        # use standard config file - single run
        config_file_path = os.path.join(
            working_dir, "config_msd_matrix_interpolation.yml"
        )
        main(config_file_path, only_phin=only_phin)
