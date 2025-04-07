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
            os.mkdir(dir)

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
        monitor="loss",
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
    phin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=msd_cfg["lr"]),
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
        # n_train = int(0.8 * x.shape[0])
        # x_train = [x[:n_train], dx_dt[:n_train], u[:n_train], mu[:n_train]]
        # x_val = [x[n_train:], dx_dt[n_train:], u[n_train:], mu[n_train:]]
        if mu is None:
            x_train = [x, dx_dt, u]
        else:
            x_train = [x, dx_dt, u, mu]
        logging.info(f"Fitting NN weights.")
        train_hist = phin.fit(
            x=x_train,
            # validation_data=(x_val, None),
            epochs=msd_cfg["n_epochs"],
            batch_size=msd_cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        save_training_times(train_hist, result_dir)
        aphin_vis.plot_train_history(train_hist, save_name=result_dir)
        phin.load_weights(os.path.join(weight_dir, ".weights.h5"))

    # write data to results directory
    save_weights(weight_dir, result_dir, load_network=msd_cfg["load_network"])
    write_to_experiment_overview(
        msd_cfg, result_dir, load_network=msd_cfg["load_network"]
    )

    # calculate all needed identified results
    msd_data_id = PHIdentifiedDataset.from_identification(msd_data, system_layer, phin)
    return msd_data_id, system_layer, phin


def main(config_path_to_file=None, only_usual_phin: bool = False):
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
    # aphin_vis.setup_matplotlib(msd_cfg["setup_matplotlib"])

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

    _, idx = np.unique(msd_data.Mu, axis=0, return_index=True)
    mu_all = msd_data.Mu[np.sort(idx), :]
    hull_convex = ConvexHull(mu_all)
    idx_train = hull_convex.vertices  # parameters that define the convex hull
    idx_test = np.setdiff1d(
        np.arange(mu_all.shape[0]), idx_train
    )  # parameters inside the convex hull
    assert idx_test.shape[0] >= 3  # at least 3 test trajectories

    # train_idx = np.array(
    #     [
    #         (i_same_mu_scenario) * 3 + np.array([0, 1, 2])
    #         for i_same_mu_scenario in idx_train
    #     ]
    # ).flatten()
    # test_idx = np.array(
    #     [
    #         (i_same_mu_scenario) * 3 + np.array([0, 1, 2])
    #         for i_same_mu_scenario in idx_test
    #     ]
    # ).flatten()
    train_idx = idx_train
    test_idx = idx_test
    shift_to_train = 45
    train_idx = np.concatenate([train_idx, test_idx[:shift_to_train]])
    test_idx = test_idx[shift_to_train:]

    # train_idx = np.arange(30)
    # test_idx = np.arange(30, 36)

    # split into train and test data
    msd_data.train_test_split_sim_idx(sim_idx_train=train_idx, sim_idx_test=test_idx)
    # scale data
    msd_data.scale_Mu(desired_bounds=msd_cfg["desired_bounds"])
    msd_data.states_to_features()

    # usual PHIN framework with mu DNN
    msd_data_id, system_layer_interp, phin_interp = phin_learning(
        msd_data,
        msd_cfg,
        configuration.directories,
        dir_extension="usual_phin ",
        layer=msd_cfg["ph_layer"],
    )

    result_dir_usual_phin = f"{result_dir}_usual_phin"
    if not os.path.isdir(result_dir_usual_phin):
        os.mkdir(result_dir_usual_phin)
    use_train_data = False
    idx_gen = "rand"
    msd_data.calculate_errors(msd_data_id)
    logging.info(f"error z: {msd_data.TRAIN.latent_error_mean}/{msd_data.TEST.latent_error_mean}")
    logging.info(f"error x: {msd_data.TRAIN.state_error_mean}/{msd_data.TEST.state_error_mean}")

    aphin_vis.plot_time_trajectories_all(
        msd_data, msd_data_id, use_train_data, idx_gen, result_dir_usual_phin
    )
    aphin_vis.plot_errors(
        msd_data,
        use_train_data,
        save_name=os.path.join(result_dir_usual_phin, "rms_error"),
    )
    aphin_vis.single_parameter_space_error_plot(
        msd_data.TEST.state_error_list[0],
        msd_data.TEST.Mu,
        msd_data.TEST.Mu_input,
        parameter_names=["mass", "stiff", "omega", "delta"],
        save_name="",
    )

    # 3d plot of initial conditions
    fig = plt.figure()
    x0 = msd_data.TRAIN.X[:,0,:,0]
    x0_test = msd_data.TEST.X[:,0,:,0]
    plt.scatter(x0[:,0], x0[:,1])
    plt.scatter(x0_test[:,0], x0_test[:,1])
    plt.xlabel("position mass 1")
    plt.ylabel("position mass 2")
    # plt.view([90,0])
    # ax.set_zlim(-0.002, 0.002)
    plt.tight_layout()
    plt.show()
    plt.savefig('initial_conditions')

    # plot of parameters
    fig = plt.figure()
    plt.plot(msd_data.TRAIN.Mu[:,0], msd_data.TRAIN.Mu[:,1], "o")
    plt.plot(msd_data.TEST.Mu[:, 0], msd_data.TEST.Mu[:, 1], "o")
    plt.xlabel("k")
    plt.ylabel("c")
    plt.tight_layout()
    plt.show()

    if only_usual_phin:
        return

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
        num_train_scenarios_with_same_mu = train_idx.shape[0] // 3
        msd_data_orig = copy.deepcopy(msd_data)
        scenario_list = []
        for i_same_mu_scenario in range(num_train_scenarios_with_same_mu):

            msd_data = copy.deepcopy(msd_data_orig)
            # there are always 3 scenarios with the same mu but different inputs
            sim_idx = (i_same_mu_scenario) * 3 + np.array([0, 1, 2])
            msd_data.decrease_num_simulations(sim_idx=sim_idx)
            mu_orig = msd_data.TRAIN.Mu[0, :]
            msd_data.remove_mu()
            msd_data_id_interp_scenario, system_layer_interp, phin_interp = (
                phin_learning(
                    msd_data,
                    msd_cfg,
                    configuration.directories,
                    dir_extension=f"interp{i_same_mu_scenario}",
                    layer=msd_cfg["layer"],
                )
            )
            scenario_list.append(
                (
                    msd_data_id_interp_scenario,
                    system_layer_interp,
                    phin_interp,
                    msd_data,
                    mu_orig,
                )
            )

        list_matrices = []
        parameter_array = np.zeros((len(scenario_list), msd_data_orig.TRAIN.n_mu))
        A_all = np.zeros((msd_data.TRAIN.n_f, msd_data.TRAIN.n_f, len(scenario_list)))
        Q_all = np.zeros((msd_data.TRAIN.n_f, msd_data.TRAIN.n_f, len(scenario_list)))
        J_all = np.zeros((msd_data.TRAIN.n_f, msd_data.TRAIN.n_f, len(scenario_list)))
        R_all = np.zeros((msd_data.TRAIN.n_f, msd_data.TRAIN.n_f, len(scenario_list)))
        B_all = np.zeros((msd_data.TRAIN.n_f, msd_data.TRAIN.n_u, len(scenario_list)))
        for i_scenario, scenario_tuple in enumerate(scenario_list):
            system_layer_interp = scenario_tuple[1]
            msd_data_id_interp_scenario = scenario_tuple[0]
            mu_orig = scenario_tuple[4]
            if msd_cfg["layer"] == "ph_layer":
                J, R, B = system_layer_interp.get_system_matrices(
                    None, msd_data_id_interp_scenario.TRAIN.n_t
                )
                Q = np.eye(J.shape[1])
            elif msd_cfg["layer"] == "phq_layer":
                J, R, B, Q = system_layer_interp.get_system_matrices(
                    None, msd_data_id_interp_scenario.TRAIN.n_t
                )
            J_all[:, :, i_scenario] = J
            R_all[:, :, i_scenario] = R
            B_all[:, :, i_scenario] = B
            A_all[:, :, i_scenario] = (J - R) @ Q
            # list_matrices.append(A)
            parameter_array[i_scenario, :] = mu_orig
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

    _, idx = np.unique(msd_data_orig.TEST.Mu, axis=0, return_index=True)
    parameter_test = msd_data_orig.TEST.Mu[np.sort(idx), :]

    plot_convex_hull = False
    if plot_convex_hull:
        hull = ConvexHull(parameter_array)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            parameter_array[:, 0], parameter_array[:, 1], parameter_array[:, 2], "b"
        )
        ax.scatter(
            parameter_test[:, 0], parameter_test[:, 1], parameter_test[:, 2], "r"
        )
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(
                parameter_array[s, 0],
                parameter_array[s, 1],
                parameter_array[s, 2],
                "r-",
            )
        plt.show()

    # %% Interpolate matrices
    weighting_array = get_weighting_function_values(
        parameter_array, parameter_test, ansatz=msd_cfg["ansatz"]
    )
    # TODO: compare PH vs LTI
    if msd_cfg["matrix_type"] == "lti":
        A_interp = evaluate_matrices(A_all, weighting_array)
        B_interp = evaluate_matrices(B_all, weighting_array)
    elif msd_cfg["matrix_type"] == "ph":
        J_interp = evaluate_matrices(J_all, weighting_array)
        R_interp = evaluate_matrices(R_all, weighting_array)
        Q_interp = evaluate_matrices(Q_all, weighting_array)
        B_interp = evaluate_matrices(B_all, weighting_array)

    # %% Create data file
    latent_shape = (
        msd_data_orig.TEST.n_sim,
        msd_data_orig.TEST.n_t,
        system_layer_interp.r,
    )
    Z_ph = np.zeros(latent_shape)
    Z_dt_ph = np.zeros(latent_shape)
    for i_test in range(weighting_array.shape[1]):
        sim_indices = (i_test) * 3 + np.array([0, 1, 2])
        if msd_cfg["matrix_type"] == "lti":
            system_lti_or_ph = LTISystem(
                A=A_interp[:, :, i_test], B=B_interp[:, :, i_test]
            )
        elif msd_cfg["matrix_type"] == "ph":
            if msd_cfg["layer"] == "phq_layer":
                system_lti_or_ph = PHSystem(
                    J_ph=J_interp[:, :, i_test],
                    R_ph=R_interp[:, :, i_test],
                    Q_ph=Q_interp[:, :, i_test],
                    B=B_interp[:, :, i_test],
                )
            if msd_cfg["layer"] == "ph_layer":
                system_lti_or_ph = PHSystem(
                    J_ph=J_interp[:, :, i_test],
                    R_ph=R_interp[:, :, i_test],
                    B=B_interp[:, :, i_test],
                )
        for i_sim in sim_indices:
            u = msd_data_orig.TEST.U[i_sim]
            msd_data_orig.TEST.get_initial_conditions()
            x_init = np.expand_dims(msd_data_orig.TEST.x_init[i_sim, :], axis=0).T
            Z_ph[i_sim], Z_dt_ph[i_sim] = system_lti_or_ph.solve_dt(
                msd_data_orig.TEST.t,
                x_init,
                u,
            )
    z_ph, dz_dt_ph = reshape_states_to_features(Z_ph, Z_dt_ph)
    x_ph, dx_dt_ph = z_ph, dz_dt_ph
    X_ph, X_dt_ph = reshape_features_to_states(
        x_ph,
        msd_data_orig.TEST.n_sim,
        msd_data_orig.TEST.n_t,
        x_dt=dx_dt_ph,
        n_n=msd_data_orig.TEST.n_n,
        n_dn=msd_data_orig.TEST.n_dn,
    )
    z = msd_data_orig.TEST.x
    Z = reshape_features_to_states(
        z, msd_data_orig.TEST.n_sim, msd_data_orig.TEST.n_t, n_f=system_layer_interp.r
    )
    z_dt = msd_data_orig.TEST.dx_dt
    Z_dt = reshape_features_to_states(
        z_dt,
        msd_data_orig.TEST.n_sim,
        msd_data_orig.TEST.n_t,
        n_f=system_layer_interp.r,
    )
    msd_data_id_interp = PHIdentifiedDataset()
    msd_data_id_interp.TEST = PHIdentifiedData(
        t=msd_data_orig.TEST.t,
        X=X_ph,
        X_dt=X_dt_ph,
        Z=Z,
        Z_dt=Z_dt,
        Z_ph=Z_ph,
        Z_dt_ph=Z_dt_ph,
        Mu=msd_data_orig.TEST.Mu,
    )
    msd_data_id_interp.TEST.states_to_features()

    use_train_data = False
    idx_gen = "rand"

    # %% get results
    result_dir_interp = f"{result_dir}_interp"
    if not os.path.isdir(result_dir_interp):
        os.mkdir(result_dir_interp)
    msd_data_orig.TEST.calculate_errors(msd_data_id_interp.TEST)
    aphin_vis.plot_time_trajectories_all(
        msd_data_orig, msd_data_id_interp, use_train_data, idx_gen, result_dir_interp
    )
    aphin_vis.plot_errors(
        msd_data_orig,
        use_train_data,
        save_name=os.path.join(result_dir_interp, "rms_error"),
    )

    # avoid that the script stops and keep the plots open
    plt.show()

    print("debug breakpoint")

    msd_data_jonas = Dataset.from_data(
        "/scratch/tmp/jrettberg/Projects/ApHIN_Review/ApHIN/examples/mass_spring_damper/data/MSD_Qeye_ph_input_siso.npz"
    )
    msd_data_jonas.train_test_split(test_size=0.06, seed=1)
    msd_data_jonas.states_to_features()

    import plotly.graph_objects as go

    trajectory = 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(msd_data.TRAIN.x.shape[0]),
            y=msd_data.TRAIN.x[:, trajectory],
            mode="lines",
            name="msd_interp",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(msd_data_jonas.TRAIN.x.shape[0]),
            y=msd_data_jonas.TRAIN.x[:, trajectory],
            mode="lines",
            name="msd_jonas",
        )
    )
    fig.show()


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
    if calc_various_experiments:
        logging.info(f"Multiple simulation runs...")
        # Run multiple simulation runs defined by parameter_variavation_dict
        configuration = Configuration(working_dir)
        _, log_dir, _, result_dir = configuration.directories

        only_usual_phin = True
        run_various_experiments(
            experiment_main_script=main,  # main without parentheses
            parameter_variation_dict=create_variation_of_parameters(),
            basis_config_yml_path=os.path.join(
                os.path.dirname(__file__), "config_msd_matrix_interpolation.yml"
            ),
            result_dir=result_dir,
            log_dir=log_dir,
            force_calculation=False,
            only_usual_phin=only_usual_phin,
        )
    else:
        # use standard config file - single run
        config_file_path = os.path.join(
            working_dir, "config_msd_matrix_interpolation.yml"
        )
        main(config_file_path)
