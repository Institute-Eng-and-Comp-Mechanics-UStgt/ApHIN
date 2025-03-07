import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.linalg


def check_list_matrices(list_matrices):
    if list_matrices[0].ndim == 3:
        if list_matrices[0].shape[0] != 1:
            raise ValueError(f"The matrix should not contain different simulations.")
        else:
            list_matrices = [
                np.squeeze(list_matrices[i], axis=0) for i in range(len(list_matrices))
            ]
    return list_matrices


def prepare_linear_system_equations(
    matrices: list | np.ndarray,
    parameter_array: np.ndarray,
    ansatz: str = "linear",
    evaluation=False,
):
    """
    parameter_array (n_systems x n_param)
    """
    if isinstance(matrices, np.ndarray):
        assert matrices.shape[2] == parameter_array.shape[0]
        n0 = matrices.shape[0]
        n1 = matrices.shape[1]
    elif isinstance(matrices, list):
        assert len(matrices) == parameter_array.shape[0]
        matrices = check_list_matrices(matrices)
        n0 = matrices[0].shape[0]
        n1 = matrices[0].shape[1]
    else:
        raise ValueError(f"Unknown type {type(matrices)} of the matrices.")

    n_system = parameter_array.shape[0]
    n_param = parameter_array.shape[1]

    if evaluation:
        n_system_eval = 1
    else:
        n_system_eval = n_system

    if isinstance(matrices, np.ndarray):
        system_matrices = matrices
    elif isinstance(matrices, list):
        # create matrices of size n0 x n1 x n_system
        system_matrices = np.zeros((n0, n1, n_system))
        for i_system in range(len(matrices)):
            system_matrices[:, :, i_system] = matrices[i_system]

    if ansatz == "linear":
        n_coeff = (n_param + 1) * n_system
        matrix_parameter_array = np.zeros((n0 * n1 * n_system_eval, n_coeff))
        matrix_entries_1d = np.zeros((n0 * n1 * n_system_eval, 1))
        # set up parameter entries
        parameter_entry_format = np.ones((n_coeff, n_system))
        for i_system in range(n_system):
            param_plus_one = np.ones((n_system, n_param + 1))
            param_plus_one[:, 1:] = np.repeat(
                np.expand_dims(parameter_array[i_system, :], axis=0), n_system, axis=0
            )
            parameter_entry_format[:, i_system] = param_plus_one.flatten()
    else:
        raise NotImplementedError(
            f"The ansatz {ansatz} has currently not been implemented."
        )

    # loop over columns and rows
    i_entry = 0
    for i_row in range(n0):
        for j_column in range(n1):
            for k_system in range(n_system_eval):
                matrix_entries_ij = system_matrices[i_row, j_column, :]
                matrix_entries_ij_repeated = np.repeat(
                    np.expand_dims(matrix_entries_ij, axis=1), n_param + 1, axis=1
                ).flatten()
                matrix_parameter_array[i_entry, :] = (
                    matrix_entries_ij_repeated * parameter_entry_format[:, k_system]
                )
                matrix_entries_1d[i_entry] = system_matrices[i_row, j_column, k_system]
                i_entry += 1

    if evaluation:
        return matrix_parameter_array
    else:
        return matrix_parameter_array, matrix_entries_1d


def get_coeff_values(
    matrices: list, parameter_array: np.ndarray, ansatz: str = "linear"
):
    matrix_parameter_array, matrix_entries_1d = prepare_linear_system_equations(
        matrices=matrices, parameter_array=parameter_array, ansatz=ansatz
    )

    # solve least square problem
    coeff_values, residuals, rank, sing_val = np.linalg.lstsq(
        matrix_parameter_array, matrix_entries_1d
    )

    return coeff_values


def evaluate_interpolation(
    coeff_values, training_matrices_list, parameter_array_eval, ansatz
):

    if parameter_array_eval.ndim == 1:
        parameter_array_eval = np.expand_dims(parameter_array_eval, axis=0)

    training_matrices_list = check_list_matrices(training_matrices_list)

    n_system_eval = parameter_array_eval.shape[0]
    n_system_training = len(training_matrices_list)
    n0 = training_matrices_list[0].shape[0]
    n1 = training_matrices_list[0].shape[1]

    system_matrix = np.zeros((n0, n1, n_system_eval))
    for i_system_eval in range(n_system_eval):
        parameter_array_eval_i_system = np.repeat(
            np.expand_dims(parameter_array_eval[i_system_eval, :], axis=0),
            n_system_training,
            axis=0,
        )
        matrix_parameter_array = prepare_linear_system_equations(
            matrices=training_matrices_list,
            parameter_array=parameter_array_eval_i_system,
            ansatz=ansatz,
            evaluation=True,
        )

        matrix_entries_i_system = matrix_parameter_array @ coeff_values
        system_matrix[:, :, i_system_eval] = np.reshape(
            matrix_entries_i_system, (n0, n1)
        )

    return system_matrix


def get_weighting_function_values(
    parameter_samples_train: np.ndarray,
    parameter_samples_eval: np.ndarray,
    ansatz: str = "linear",
):
    """
    parameter_samples_train: n_system x n_param array
    parameter_samples_eval: n_eval_points x n_param array
    Returns:
    array of (n_system,n_eval_points) weighting values
    """
    assert parameter_samples_train.shape[1] == parameter_samples_eval.shape[1]

    n_system = parameter_samples_train.shape[0]
    n_eval_points = parameter_samples_eval.shape[0]

    weighting_array = np.zeros((n_system, n_eval_points))
    for i_system in range(n_system):
        values = np.zeros(n_system)
        values[i_system] = 1
        weighting_array[i_system, :] = scipy.interpolate.griddata(
            points=parameter_samples_train,
            values=values,
            xi=parameter_samples_eval,
            method=ansatz,
        )

    print(
        f"Max sum difference: {np.max(np.abs(np.sum(weighting_array, axis=0)- np.ones(n_eval_points)))}"
    )
    # assert np.allclose(np.sum(weighting_array, axis=0), np.ones(n_eval_points))

    return weighting_array


def evaluate_matrices(matrices_training: np.ndarray, weighting_array: np.ndarray):
    """
    matrices_training (n0,n1,n_system): matrices gained from training parameter samples parameter_samples_train
    weighting_array (n_system,n_eval_points): usually calculated with get_weighting_function_values
    Returns:
    requested_matrices of size (n0,n1,n_eval_points)
    """
    assert matrices_training.shape[2] == weighting_array.shape[0]

    n0 = matrices_training.shape[0]
    n1 = matrices_training.shape[1]
    n_system = matrices_training.shape[2]
    n_eval_points = weighting_array.shape[1]

    requested_matrices = np.zeros((n0, n1, n_eval_points))
    for i_eval_point in range(n_eval_points):
        for i_system in range(n_system):
            requested_matrices[:, :, i_eval_point] = (
                requested_matrices[:, :, i_eval_point]
                + weighting_array[i_system, i_eval_point]
                * matrices_training[:, :, i_system]
            )
    return requested_matrices


def in_hull(points, x):
    # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = scipy.optimize.linprog(c, A_eq=A, b_eq=b)
    return lp.success

    # # matrices_conc = np.squeeze(np.concatenate((list_matrices), axis=1), axis=0)
    # # matrix_entries_1d = matrices_conc.flatten()

    # # matrix_entries_diag = np.diag(matrix_entries_1d)

    # # size_matrix = list_matrices[0].shape
    # # linear_equations = size_matrix[0] * size_matrix[1] * n_system

    # if ansatz == "linear":
    #     # initialize arrays
    #     n_coeff = (n_param + 1) * n_system
    #     matrix_parameter_array = np.zeros(
    #         (size_matrix[0] * size_matrix[1] * n_system, n_coeff)
    #     )
    #     for i_entries in range(matrix_entries_1d.shape[0]):
    #         matrix_entry = np.repeat(
    #             np.expand_dims(matrix_entries_1d[i_entries], axis=1),
    #             n_param + 1,
    #             axis=1,
    #         )
    #     # parameter values

    #     param_matrix = []
    #     for i_system in range(n_system):
    #         params_i = np.ones((size_matrix[0] * size_matrix[1], n_param + 1))
    #         params_i[:, 1:] = np.repeat(
    #             np.expand_dims(parameter_array[i_system, :], axis=0),
    #             size_matrix[0] * size_matrix[1],
    #             axis=0,
    #         )
    #         param_matrix.append(params_i)
    #     param_matrix = scipy.linalg.block_diag(*param_matrix)
    #     assert linear_equations == param_matrix[0]

    # # solve linear system of equations
    # # system of the form
    # # matrix_entries_diag@param_matrix @ coeff_values = matrix_entries_1d
    # # where matrix_entries_diag contains all matrix entries in diagonal structure, i.e. diag matrix of size n*n*n_system
    # # param_matrix uses a block-diagonal structure with constant term and parameter terms, i.e.
    # # [1 p1s1 p2s1 0 0 0|
    # # 1 p1s1 p2s1  0 0 0| repeated n*n times
    # # 1 p1s1 p2s1  0 0 0|
    # # 0 0   0 1 p1s2 p2s2 |
    # # 0 0   0 1 p1s2 p2s2 | repeated n*n times
    # # 0 0   0 1 p1s2 p2s2 |
    # # ]
    # # matrix_entries_1d contains all matrix entries in vector form
    # coeff_values, residuals, rank, sing_val = np.linalg.lstsq(
    #     matrix_entries_diag @ param_matrix, matrix_entries_1d
    # )

    # return coeff_values
