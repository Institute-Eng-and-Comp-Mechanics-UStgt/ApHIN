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
from aphin.layers.phq_layer import PHQLayer, PHLayer
from aphin.layers import DescriptorPHQLayer, DescriptorPHLayer
from aphin.utils.save_results import (
    save_weights,
    write_to_experiment_overview,
    save_evaluation_times,
    save_training_times,
)
from aphin.utils.experiments import run_various_experiments
from aphin.utils.print_matrices import print_matrices

# tf.config.run_functions_eagerly(True)


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
    manual_results_folder = "synrm_with_pca_and_dynamic"  # {None} if no results shall be loaded, else create str with folder name or path to results folder

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

    synrm_data = SynRMDataset.from_matlab(
        data_path=srm_cfg["matfile_path"], exclude_states=srm_cfg["exclude_states"]
    )

    V = SynRMDataset.from_matlab(data_path=srm_cfg["matfile_path"], return_V=True)

    aphin_vis.setup_matplotlib(srm_cfg["setup_matplotlib"])

    # filter data with savgol filter
    if srm_cfg["filter_data"]:
        logging.info("Data is filtered")
        synrm_data.filter_data(interp_equidis_t=False)
    else:
        logging.info("Data is not filtered.")

    # reduced size
    r = srm_cfg["r"]

    # train-test split
    # sim_idx_train = np.arange(10)
    # sim_idx_test = np.arange(5) + len(sim_idx_train)
    sim_idx_train = [2]
    sim_idx_test = [1]
    synrm_data.train_test_split_sim_idx(
        sim_idx_train=sim_idx_train, sim_idx_test=sim_idx_test
    )
    # synrm_data.train_test_split(test_size=0.2, seed=srm_cfg["seed"])

    # decrease number of simulations
    if srm_cfg["num_sim"] is not None:
        synrm_data.decrease_num_simulations(
            num_sim=srm_cfg["num_sim"], seed=srm_cfg["seed"]
        )

    num_rand_pick_entries = srm_cfg["num_rand_pick_entries"]
    if srm_cfg["high_dimensional"]:
        if srm_cfg["exclude_states"] == "no_rigid":
            synrm_data.reproject_with_basis(
                [V, V],
                idx=[slice(75, 95), slice(95, 115)],
                pick_method=srm_cfg["pick_method"],
                pick_entry=num_rand_pick_entries,
                seed=srm_cfg["seed"],
            )
        if srm_cfg["exclude_states"] is None:
            # all states
            if srm_cfg["pick_method"] == "idx":
                # use coarsen factor
                coarsen_factor = srm_cfg["coarsen_factor"]
                num_nodes = 121822
                coarse_node_indices = np.arange(0, num_nodes, coarsen_factor)
                dof_indices = np.zeros(coarse_node_indices.shape[0] * 3, dtype=np.int64)
                for i_node_index, coarse_node_index in enumerate(coarse_node_indices):
                    dof_indices[i_node_index * 3 : (i_node_index + 1) * 3] = np.arange(
                        coarse_node_index * 3, (coarse_node_index + 1) * 3
                    )
                dof_indices_list = list(dof_indices)

                synrm_data.reproject_with_basis(
                    [V, V],
                    idx=[slice(80, 100), slice(105, 125)],
                    pick_method=srm_cfg["pick_method"],
                    pick_entry=dof_indices_list,
                    seed=srm_cfg["seed"],
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    # TODO: Move to class
    if srm_cfg["exclude_states"] == "no_phi":
        idx_eta = np.arange(3)
        # idx_phi = np.arange(3, 75)
        idx_rigid = np.arange(3, 8)
        idx_elastic_modes = np.arange(8, 28)
        idx_Drigid = np.arange(28, 33)
        idx_Delastic = np.arange(33, 53)
        scaling_rigid = np.max(np.abs(synrm_data.TRAIN.X[..., idx_rigid]))
        scaling_Drigid = np.max(np.abs(synrm_data.TRAIN.X[..., idx_Drigid]))
        scaling_Delastic = np.max(np.abs(synrm_data.TRAIN.X[..., idx_Delastic]))
    elif srm_cfg["exclude_states"] == "no_rigid":
        idx_eta = np.arange(3)
        idx_phi = np.arange(3, 75)
        # idx_rigid = np.arange(75, 80)
        idx_elastic_modes = np.arange(75, 95)
        # idx_Drigid = np.arange(100, 105)
        idx_Delastic = np.arange(95, 115)
        scaling_phi = np.max(np.abs(synrm_data.TRAIN.X[..., idx_phi])) * 36
        scaling_Delastic = np.max(np.abs(synrm_data.TRAIN.X[..., idx_Delastic]))
        scaling_eta = np.max(np.abs(synrm_data.TRAIN.X[..., idx_eta])) * 3
    elif srm_cfg["exclude_states"] == "no_velocities":
        idx_eta = np.arange(3)
        idx_phi = np.arange(3, 75)
        idx_rigid = np.arange(75, 80)
        idx_elastic_modes = np.arange(80, 100)
        scaling_phi = np.max(np.abs(synrm_data.TRAIN.X[..., idx_phi])) * 36
        scaling_rigid = np.max(np.abs(synrm_data.TRAIN.X[..., idx_rigid]))
    elif srm_cfg["exclude_states"] == "only_elastic":
        idx_elastic_modes = np.arange(0, 20)
        idx_Delastic = np.arange(20, 40)
        scaling_elastic = np.max(np.abs(synrm_data.TRAIN.X[..., idx_elastic_modes]))
        scaling_Delastic = np.max(np.abs(synrm_data.TRAIN.X[..., idx_Delastic]))
    else:
        idx_eta = np.arange(3)
        idx_phi = np.arange(3, 75)
        idx_rigid = np.arange(75, 80)
        if srm_cfg["high_dimensional"]:
            idx_elastic_modes = np.arange(80, 80 + len(dof_indices_list))
            idx_Delastic = np.arange(105, 105 + len(dof_indices_list))
        else:
            idx_elastic_modes = np.arange(80, 100)
            idx_Delastic = np.arange(105, 125)
        idx_Drigid = np.arange(100, 105)
        scaling_eta = np.max(np.abs(synrm_data.TRAIN.X[..., idx_eta])) * 3
        scaling_phi = np.max(np.abs(synrm_data.TRAIN.X[..., idx_phi])) * 36
        scaling_rigid = np.max(np.abs(synrm_data.TRAIN.X[..., idx_rigid]))
        scaling_Drigid = np.max(np.abs(synrm_data.TRAIN.X[..., idx_Drigid]))
        scaling_Delastic = np.max(np.abs(synrm_data.TRAIN.X[..., idx_Delastic]))
    scaling_elastic = np.max(np.abs(synrm_data.TRAIN.X[..., idx_elastic_modes]))
    # scaling_rigid = np.max([scaling_rigid, scaling_Drigid])
    # scaling_elastic = np.max([scaling_elastic, scaling_Delastic])
    bounds_u = np.array(
        [np.min(synrm_data.TRAIN.U), np.max(synrm_data.TRAIN.U)]
    ) / np.max(np.abs(synrm_data.TRAIN.U))
    bounds_u = [bounds_u[0], bounds_u[1]]

    # scale data
    if srm_cfg["exclude_states"] == "no_phi":
        scaling_values = [
            scaling_eta,
            scaling_rigid,
            scaling_elastic,
            scaling_Drigid,
            scaling_Delastic,
        ]
    elif srm_cfg["exclude_states"] == "no_rigid":
        scaling_values = [
            scaling_eta,
            scaling_phi,
            scaling_elastic,
            scaling_Delastic,
        ]
    elif srm_cfg["exclude_states"] == "no_velocities":
        scaling_values = [
            scaling_eta,
            scaling_phi,
            scaling_rigid,
            scaling_elastic,
        ]
    elif srm_cfg["exclude_states"] == "only_elastic":
        scaling_values = [
            scaling_elastic,
            scaling_Delastic,
        ]
    else:
        scaling_values = [
            scaling_eta,
            scaling_phi,
            scaling_rigid,
            scaling_elastic,
            scaling_Drigid,
            scaling_Delastic,
        ]
    # synrm_data.scale_all(
    #     scaling_values=scaling_values,
    #     domain_split_vals=srm_cfg["domain_split_vals"],
    #     u_desired_bounds=bounds_u,
    #     mu_desired_bounds=srm_cfg["desired_bounds"],
    # )
    synrm_data.scale_X(
        scaling_values=scaling_values, domain_split_vals=srm_cfg["domain_split_vals"]
    )
    # scale u manually
    u_domains = [3, 36]
    start_idx = 0
    input_scaling_values = []
    for u_domain in u_domains:
        scaling_value = np.max(
            np.abs(synrm_data.TRAIN.U[:, :, start_idx : start_idx + u_domain])
        )
        synrm_data.TRAIN.U[:, :, start_idx : start_idx + u_domain] = (
            synrm_data.TRAIN.U[:, :, start_idx : start_idx + u_domain] / scaling_value
        )
        synrm_data.TEST.U[:, :, start_idx : start_idx + u_domain] = (
            synrm_data.TEST.U[:, :, start_idx : start_idx + u_domain] / scaling_value
        )
        input_scaling_values.append(scaling_value)
        start_idx += u_domain
    synrm_data.reshape_inputs_to_features()

    # transform to feature form that is used by the deep learning
    synrm_data.states_to_features()
    t, x, dx_dt, u, mu = synrm_data.data

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
        tensorboard=srm_cfg["tensorboard"],
        log_dir=log_dir,
        monitor=monitor,
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
        if srm_cfg["prelearn_rec"]:
            logging.info(f"Loading NN weights from only rec learning.")
            aphin.load_weights(os.path.join(result_dir, ".weights.h5"))
        logging.info(f"Fitting NN weights.")
        n_train = int(0.8 * x.shape[0])
        x_train = [x[:n_train], dx_dt[:n_train], u[:n_train]]
        x_val = [x[n_train:], dx_dt[n_train:], u[n_train:]]
        train_hist = aphin.fit(
            x=x_train,
            validation_data=x_val,
            epochs=srm_cfg["n_epochs"],
            batch_size=srm_cfg["batch_size"],
            verbose=2,
            callbacks=callback,
        )
        save_training_times(train_hist, result_dir)
        aphin_vis.plot_train_history(
            train_hist, save_path=result_dir, validation=validation
        )

        # load best weights
        aphin.load_weights(os.path.join(weight_dir, ".weights.h5"))

    # write data to results directory
    save_weights(weight_dir, result_dir, load_network=srm_cfg["load_network"])
    write_to_experiment_overview(
        srm_cfg, result_dir, load_network=srm_cfg["load_network"]
    )

    # %% Validation
    logging.info(
        "################################   3. Validation ################################"
    )
    t_test, x_test, dx_dt_test, u_test, mu_test = synrm_data.test_data
    n_sim_test, n_t_test, _, _, _, _ = synrm_data.shape_test
    print_matrices(system_layer, mu=mu_test, n_t=n_t_test)

    # calculate projection and Jacobian errors
    # file_name = "projection_error.txt"
    # projection_error_file_dir = os.path.join(result_dir, file_name)
    # aphin.get_projection_properties(x, x_test, file_dir=projection_error_file_dir)

    # %% Validation of the AE reconstruction
    # get original quantities
    synrm_data_id = PHIdentifiedDataset.from_identification(
        synrm_data, system_layer, aphin, integrator_type="imr"
    )
    save_evaluation_times(synrm_data_id, result_dir)

    e_rec = np.linalg.norm(
        synrm_data.TEST.X - synrm_data_id.TEST.X_rec
    ) / np.linalg.norm(synrm_data.TEST.X)
    e_z = np.linalg.norm(
        synrm_data_id.TEST.Z - synrm_data_id.TEST.Z_ph
    ) / np.linalg.norm(synrm_data_id.TEST.Z)
    e_z_dt = np.linalg.norm(
        synrm_data_id.TEST.Z_dt - synrm_data_id.TEST.Z_dt_ph
    ) / np.linalg.norm(synrm_data_id.TEST.Z_dt)
    e_z_dt_map = np.linalg.norm(
        synrm_data_id.TEST.Z_dt - synrm_data_id.TEST.Z_dt_ph_map
    ) / np.linalg.norm(synrm_data_id.TEST.Z_dt)
    e_x = np.linalg.norm(synrm_data.TEST.X - synrm_data_id.TEST.X) / np.linalg.norm(
        synrm_data.TEST.X
    )

    # reproject
    # num_rand_pick_entries = 1000
    # synrm_data.reproject_with_basis(
    #     [V, V],
    #     idx=[slice(80, 100), slice(105, 125)],
    #     pick_method="rand",
    #     pick_entry=num_rand_pick_entries,
    #     seed=srm_cfg["seed"],
    # )
    # synrm_data_id.reproject_with_basis(
    #     [V, V],
    #     idx=[slice(80, 100), slice(105, 125)],
    #     pick_method="rand",
    #     pick_entry=num_rand_pick_entries,
    #     seed=srm_cfg["seed"],
    # )

    # domain_split_vals_projected = [
    #     3,
    #     72,
    #     5,
    #     num_rand_pick_entries,
    #     5,
    #     num_rand_pick_entries,
    # ]

    # synrm_data.calculate_errors(
    #     synrm_data_id,
    #     domain_split_vals=domain_split_vals_projected,
    #     save_to_txt=True,
    #     result_dir=result_dir,
    # )

    # %% calculate errors
    synrm_data.calculate_errors(
        synrm_data_id,
        domain_split_vals=srm_cfg["domain_split_vals"],
        save_to_txt=True,
        result_dir=result_dir,
    )
    aphin_vis.plot_errors(
        synrm_data,
        t=synrm_data.t_test,
        save_name=os.path.join(result_dir, "rms_error"),
        domain_names=srm_cfg["domain_names"],
        save_to_csv=True,
        yscale="log",
    )

    # %% plot trajectories
    use_train_data = True
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
        synrm_data,
        synrm_data_id,
        use_train_data=use_train_data,
        idx_gen=idx_gen,
        result_dir=result_dir,
        idx_custom_tuple=idx_custom_tuple,
    )

    aphin_vis.plot_u(data=synrm_data, use_train_data=use_train_data)

    # avoid that the script stops and keep the plots open
    # plt.show()

    # print("debug")


# parameter variation for multiple experiment runs
# requires calc_various_experiments = True
def create_variation_of_parameters():
    parameter_variation_dict = {
        "l_rec": [1],
        "l_dz": [0.001],
        "l_dx": [0, 0.000001],
        "r": [20, 40, 60],
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
