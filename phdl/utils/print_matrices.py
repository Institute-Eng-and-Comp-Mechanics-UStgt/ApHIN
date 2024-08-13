import numpy as np
import logging


def matprint(name, mat, decimals=2):
    """
    Pretty print a matrix with specified formatting.

    This function prints the matrix with a label and formats the floating-point numbers
    to a specified number of decimal places for better readability.

    Parameters:
    -----------
    name : str
        Label to display before printing the matrix.
    mat : numpy.ndarray
        The matrix to be printed.
    decimals : int, optional
        Number of decimal places to format the floating-point numbers. Defaults to 2.

    Examples:
    ---------
    >>> mat = np.array([[1.123456, 2.345678], [3.456789, 4.567890]])
    >>> matprint("My Matrix", mat, decimals=3)
    My Matrix:
    [[1.123 2.346]
     [3.457 4.568]]
    """
    print(f"\n{name}:")
    with np.printoptions(
        precision=4,
        suppress=False,
        formatter={"float": ("{:0." + str(decimals) + "f}").format},
        linewidth=100,
    ):
        print(mat)


def print_matrices(
    layer, mu=None, n_t=None, sim_idx=0, data=None, decimals=2, use_train_data=False
):
    """
    Print system matrices and their original counterparts, if available.

    This function extracts and prints the system matrices (J, R, B, Q) from a given layer
    and compares them with their original counterparts if provided. It also logs the minimum
    eigenvalues of matrices Q and R.

    Parameters:
    -----------
    layer : instance of LTILayer or its children
        An object that provides the `get_system_matrices` method to retrieve the system matrices.
    mu : array-like, optional
        Parameter values. Defaults to None.
    n_t : int or None, optional
        Number of time steps. Defaults to None.
    sim_idx : int, optional
        Simulation index for selecting specific matrices from a list of matrices. Defaults to 0.
    data : object from PHIdentifiedDataset or None, optional
        An object containing original matrices for comparison. Defaults to None.
    decimals : int, optional
        Number of decimal places to format the matrix values. Defaults to 2.
    use_train_data : bool, optional
        If True, uses training data for original matrices; otherwise, uses test data. Defaults to False.
    """
    # identified matrices
    matrix_names = ["J", "R", "B", "Q"]  # standard order
    matrices = list(layer.get_system_matrices(mu, n_t))
    if data is not None:
        # original matrices
        if use_train_data:
            matrices_orig = list(data.ph_matrices)
        else:
            matrices_orig = list(data.ph_matrices_test)
        # swap position of Q and B
        matrices_orig[2], matrices_orig[3] = matrices_orig[3], matrices_orig[2]
        matrix_names_orig = ["J_orig", "R_orig", "B_orig", "Q_orig"]

    for i, (name, mat) in enumerate(zip(matrix_names[: len(matrices)], matrices)):
        if mat is None:
            # e.g. no input matrix B; will be skipped
            continue
        mat = mat[sim_idx]
        matprint(name, mat, decimals=decimals)
        if data is not None:
            if matrices_orig[i] is None:
                raise ValueError("Original matrix is required in the data instance.")
            mat_orig = matrices_orig[i][sim_idx]
            matrix_name_orig = matrix_names_orig[i]
            matprint(matrix_name_orig, mat_orig, decimals=decimals)

    if len(matrices) == 4:
        eigvals_Q_list = [
            np.min(np.linalg.eigvals(matrices[3][i]))
            for i in range(matrices[3].shape[0])
        ]
        logging.info(f"Minimum eigenvalue of Q: {np.min(eigvals_Q_list)}")

    # calculate all minimal eigenvalues
    eigvals_R_list = [
        np.min(np.linalg.eigvals(matrices[1][i])) for i in range(matrices[1].shape[0])
    ]
    logging.info(f"Minimum eigenvalue of all R: {np.min(eigvals_R_list)}")
