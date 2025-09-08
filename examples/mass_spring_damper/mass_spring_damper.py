# default packages
import logging
import os
import numpy as np
import copy
import sys

# third party packages
import tensorflow as tf
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

# Add the system directory to sys.path
system_dir = os.path.dirname(os.path.abspath(__file__))
if system_dir not in sys.path:
    sys.path.append(system_dir)
from data_generation.matrix_interpolation import (
    get_weighting_function_values,
    evaluate_matrices,
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
        earlystopping=False,
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
    aphin_vis.setup_matplotlib(save_plots=msd_cfg["save_plots"])

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
    TEST_or_TRAIN = "TEST" if use_train_data is False else "TRAIN"
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

    # permute matrices to fit the publication
    perm = [0, 3, 1, 4, 2, 5]
    msd_data.permute_matrices(permutation_idx=perm)
    # test_ids = [0, 1, 3, 6, 7]  # test_ids = range(10) # range(6) test_ids = [0]
    rng = np.random.default_rng(seed=msd_cfg["seed"])
    test_ids = rng.integers(0, msd_data.TEST.n_sim, size=(5,))

    create_costum_plot = True
    if create_costum_plot:
        rng = np.random.default_rng()
        n_n_array = np.arange(3)
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
            save_to_csv=False,
            save_name="msd_custom_nodes_phin",
        )

    if only_phin:
        return

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%                        LTI                                             %%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dir_extension = "lti"
    msd_data_id_lti, system_layer_lti, _ = phin_learning(
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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%                        Matrix interpolation (MI)                       %%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    # test with sample data:
    test_with_sample_data = False
    if test_with_sample_data:
        weighting_array = get_weighting_function_values(
            parameter_array, parameter_array, ansatz=msd_cfg["ansatz"]
        )
        matrices_eval = evaluate_matrices(A_all, weighting_array)
        print(np.allclose(A_all, matrices_eval))  # should be True

    for TEST_or_TRAIN in ["TRAIN", "TEST"]:
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
        system_lti_or_ph = []
        for i_test in range(weighting_array.shape[1]):
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

    result_dir_mi = os.path.join(result_dir, "mi", "test")
    use_train_data = False

    os.makedirs(result_dir_mi, exist_ok=True)

    msd_data_orig.calculate_errors(
        msd_data_id_mi,
        domain_split_vals=[1, 1],
        save_to_txt=True,
        result_dir=result_dir_mi,
    )

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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%                        Save data for paper figures                     %%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ref_dir = os.path.join(result_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    # state data
    for data_, dir_ in zip(
        [msd_data, msd_data_id_lti, msd_data_id_mi, msd_data_id_phin],
        [ref_dir, result_dir_lti, result_dir_mi, result_dir_phin],
    ):

        getattr(data_, TEST_or_TRAIN).calculate_eigenvalues(
            result_dir=dir_,
            save_to_csv=True,
            save_name="eigenvalues",
        )

        for dof in range(3):
            getattr(data_, TEST_or_TRAIN).save_state_traj_as_csv(
                dir_,
                second_oder=True,
                dof=dof,
                filename=f"state_{dof}_trajectories",
            )

        # chessboard visualisation
        J_pred, R_pred, Q_pred, B_pred = getattr(data_, TEST_or_TRAIN).ph_matrices
        # only implemented for test case
        aphin_vis.chessboard_visualisation(
            test_ids,
            msd_data_orig,
            matrices_pred=(
                (J_pred, R_pred, B_pred, Q_pred)
                if isinstance(Q_pred, np.ndarray)
                else (J_pred, R_pred, B_pred)
            ),
            result_dir=dir_,
            limits=msd_cfg["matrix_color_limits"],
            error_limits=msd_cfg["matrix_error_limits"],
        )

    # avoid that the script stops and keep the plots open
    # plt.show()


def create_variation_of_parameters():
    parameter_variation_dict = {"example_config_key": ["value1", "value2"]}
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
            basis_config_yml_path=os.path.join(os.path.dirname(__file__), "config.yml"),
            result_dir=result_dir,
            log_dir=log_dir,
            only_phin=only_phin,
        )
    else:
        # use standard config file - single run
        config_file_path = os.path.join(working_dir, "config.yml")
        main(config_file_path, only_phin=only_phin)
