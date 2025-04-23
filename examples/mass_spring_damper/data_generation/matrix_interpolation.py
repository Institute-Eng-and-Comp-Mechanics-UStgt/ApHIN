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


def get_weighting_function_values(
    parameter_samples_train: np.ndarray,
    parameter_samples_eval: np.ndarray,
    ansatz: str = "linear",
):
    """
    Computes weighting function values for interpolation of parameter-dependent models.

    For each training parameter sample, a weighting function is constructed over a set of
    evaluation parameter points using interpolation. The output can be interpreted as
    barycentric or interpolation weights indicating the influence of each training point
    at each evaluation point.

    Parameters:
    -----------
    parameter_samples_train : np.ndarray
        Array of shape (n_system, n_param) containing parameter samples used during training.

    parameter_samples_eval : np.ndarray
        Array of shape (n_eval_points, n_param) with the evaluation points where weighting
        values should be computed.

    ansatz : str, optional
        Interpolation method to use. Supported options are "linear", "cubic", and "nearest".
        Note that "cubic" is only supported for 1D and 2D parameter spaces.

    Returns:
    --------
    np.ndarray
        Array of shape (n_system, n_eval_points) containing the computed weighting values.
        Each column sums to 1 across systems, indicating a convex combination.

    Raises:
    -------
    AssertionError
        If input shapes are inconsistent or if unsupported conditions for "cubic" interpolation are met.

    Notes:
    ------
    Uses `scipy.interpolate.griddata` to perform the interpolation.
    The function asserts that the sum of weights for each evaluation point is approximately 1.
    """
    assert parameter_samples_train.shape[1] == parameter_samples_eval.shape[1]
    assert ansatz in ["linear", "cubic", "nearest"]
    if ansatz == "cubic":
        assert parameter_samples_train.shape[1] <= 2  # method only exists for 1D and 2D

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
    assert np.allclose(np.sum(weighting_array, axis=0), np.ones(n_eval_points))

    return weighting_array


def evaluate_matrices(matrices_training: np.ndarray, weighting_array: np.ndarray):
    """
    Evaluates interpolated matrices at given parameter evaluation points using a set of
    weighting values.

    This function combines training matrices based on a weighting scheme, typically obtained
    from an interpolation method such as `get_weighting_function_values`, to estimate
    parameter-dependent matrices at new evaluation points.

    Parameters:
    -----------
    matrices_training : np.ndarray
        Array of shape (n_system, n0, n1) containing matrices associated with training
        parameter samples. Each matrix corresponds to one system or training sample.

    weighting_array : np.ndarray
        Array of shape (n_system, n_eval_points) containing weighting values for combining
        the training matrices. Each column represents weights for one evaluation point and
        should sum to 1.

    Returns:
    --------
    np.ndarray
        Array of shape (n_eval_points, n0, n1) containing the interpolated or reconstructed
        matrices at the evaluation points.

    Raises:
    -------
    AssertionError
        If the number of systems in `matrices_training` and `weighting_array` do not match.

    Notes:
    ------
    The resulting matrices are a convex combination of the training matrices if the weights
    are non-negative and sum to one.
    """
    assert matrices_training.shape[0] == weighting_array.shape[0]

    n0 = matrices_training.shape[1]
    n1 = matrices_training.shape[2]
    n_system = matrices_training.shape[0]
    n_eval_points = weighting_array.shape[1]

    requested_matrices = np.zeros((n_eval_points, n0, n1))
    for i_eval_point in range(n_eval_points):
        for i_system in range(n_system):
            requested_matrices[i_eval_point, :, :] = (
                requested_matrices[i_eval_point, :, :]
                + weighting_array[i_system, i_eval_point]
                * matrices_training[i_system, :, :]
            )
    return requested_matrices


def in_hull(points, x):
    """
    Determines whether a point lies inside the convex hull of a given set of points.

    This function solves a linear programming problem to check if the point `x` can be
    represented as a convex combination of the rows in `points`, which implies that `x` lies
    within the convex hull formed by those points.

    From: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl

    Parameters:
    -----------
    points : np.ndarray
        An array of shape (n_points, n_dim) representing the set of points forming the convex hull.

    x : np.ndarray
        A 1D array of shape (n_dim,) representing the point to test for inclusion in the convex hull.

    Returns:
    --------
    bool
        True if `x` lies within the convex hull of `points`, False otherwise.

    Notes:
    ------
    This implementation uses `scipy.optimize.linprog` to determine whether the point lies
    within the convex hull. It assumes the input dimensions are consistent.
    """
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = scipy.optimize.linprog(c, A_eq=A, b_eq=b)
    return lp.success
