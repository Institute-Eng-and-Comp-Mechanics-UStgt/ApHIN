import numpy as np
import logging
from scipy import linalg

from pymor.models.iosys import PHLTIModel, LTIModel


def transform_pH_to_Q_identity(J, R, Q, B, C, solver="Q", seed=1):
    """
    Transform a port-Hamiltonian (pH) system to one with Q = I (identity matrix).

    This function transforms a given port-Hamiltonian system to a new representation where the matrix `Q` is replaced
    by the identity matrix. The transformation is described in [BeattieMehrmannVanDooren18] and involves solving a
    Riccati equation or using `Q` directly as a solution to the KYP (Kalman-Yakubovich-Popov) inequality.

    Parameters
    ----------
    J : array-like, shape (n, n)
        The port-Hamiltonian J matrix, which should be skew-symmetric.
    R : array-like, shape (n, n)
        The port-Hamiltonian R matrix, which should be symmetric positive semi-definite.
    Q : array-like, shape (n, n)
        The port-Hamiltonian Q matrix to be transformed to the identity matrix.
    B : array-like, shape (n, n_u)
        The input matrix of the system.
    C : array-like, shape (n_u, n)
        The output matrix of the system.
    solver : {'scipy', 'pymor', 'Q'}, optional
        Method for solving the transformation:
        - 'scipy' or 'pymor': Use the Riccati equation solver.
        - 'Q': Use the provided Q matrix directly.
    seed : int, optional
        Random seed for generating a dummy input matrix when the Riccati equation is solved.

    Returns
    -------
    J_T : ndarray, shape (n, n)
        The transformed J matrix.
    R_T : ndarray, shape (n, n)
        The transformed R matrix.
    B_T_ph : ndarray, shape (n, n_u)
        The transformed input matrix.
    C_T_ph : ndarray, shape (n_u, n)
        The transformed output matrix.
    T : ndarray, shape (n, n)
        The transformation matrix.
    T_inv : ndarray, shape (n, n)
        The inverse of the transformation matrix.
    input_used : bool
        Whether the input matrix was used in the transformation.
    Qeye_system_exists : bool
        Whether a valid transformation to Q = I was found.
    """

    second_order_size = int(J.shape[0] / 2)
    solver_list = ["scipy", "pymor", "Q"]
    assert solver in solver_list

    # calculate system matrix
    A_ph = (J - R) @ Q
    # no feedthrough
    D = np.zeros((B.shape[1], B.shape[1]))

    # autonomous system?
    if B.sum() == 0:
        input_used = False
    else:
        input_used = True

    if solver in ["scipy", "pymor"]:
        if not (input_used):
            # no input
            # solving the Riccatti equation relies on an input matrix
            # if the pH system is autonomous (dissipative Hamiltonian system), we use a dummy input and output to solve the Riccatti equation

            # dummy B matrix either with one entry as 1 or as random matrix (random seems to work better)
            B_as_random = True
            if B_as_random:
                # B as random entries (only second half of state)
                rng = np.random.default_rng(seed)
                rand_indices_B = rng.integers(
                    second_order_size, 2 * second_order_size, size=second_order_size
                )
                B_Ricc = np.zeros(2 * second_order_size)[:, np.newaxis]
                B_Ricc[rand_indices_B] = rng.random(second_order_size)[:, np.newaxis]
            else:
                # B is one entry 1
                B_Ricc = np.zeros(2 * second_order_size)[:, np.newaxis]
                B_Ricc[second_order_size] = 1

            C = B_Ricc.T

        else:
            B_Ricc = B.copy()
            C = C.copy()

        # use small identity for feedthrough
        epsilon = 1e-12
        D = epsilon * np.eye(B_Ricc.shape[1])
        checkPR(A_ph, B_Ricc, C, D)

        X, Qeye_system_exists = solve_Riccati(A_ph, B_Ricc, C, D, solver)
    elif solver == "Q":
        Qeye_system_exists = True
        # use Q as solution to the KYP inequality
        logging.info(f"Using Q for transformation to identity.")
        X = Q
    else:
        raise ValueError(
            f"Unknown value for solver {solver}. Choose from valid entries: {solver_list}"
        )

    J_T, R_T, B_T_ph, C_T_ph, T, T_inv = Q_to_I_transformation(
        Qeye_system_exists, A_ph, B, C, D, X
    )

    return J_T, R_T, B_T_ph, C_T_ph, T, T_inv, input_used, Qeye_system_exists


def checkPR(A_ph, B, C, D):
    """
    Check if a system is positive-real.

    This function evaluates the positive-realness of a linear system defined by the matrices \(A_{ph}\), \(B\), \(C\), and \(D\).
    It checks if the system's transfer function matrix is positive semidefinite for several frequencies on the imaginary axis.

    Parameters:
    -----------
    A_ph : numpy.ndarray
        System matrix of the system.
    B : numpy.ndarray
        Input matrix of the system.
    C : numpy.ndarray
        Output matrix of the system.
    D : numpy.ndarray
        Feedthrough matrix of the system.

    Notes:
    ------
    - Positive-realness requires that \(\Phi(s) = T(-s)^\dagger + T(s)\) is positive semidefinite for all \(\omega\) in \(\mathbb{R}\).
    - The function checks this for 100 points between 0.1 and 1000 rad/s.
    - For stability, it is sufficient to check if the (1,1) block of \(W(X)\) is positive semidefinite.
    """
    # Check for positive-real systems
    calT = lambda s: D + C @ np.linalg.inv(s * np.eye(A_ph.shape[0]) - A_ph) @ B
    Phi = lambda s: (calT(-s)).conj().T + calT(s)
    # Phi needs to be positive semidefinite for all omega in i*R (on imaginary axis)
    # (checked for 100 points between 0.1 and 1000)
    omega_vec = np.linspace(0.1, 1000, 100)
    negEigValPhi = False
    for omega in omega_vec:
        eig_vals_Phi, eig_vectors_Phi = linalg.eig(Phi(1j * omega))
        if (eig_vals_Phi < 0).any():
            print(f"negative eigenvalues at i{omega} rad/s \n")
            negEigValPhi = True
    if not negEigValPhi:
        print(f"Phi is positive semidefinite - X that satisfies W(X) should exist")

    # X itself needs to pos. def. for the system to be stable (Theorem 1)
    # However, it is sufficient for (asymptotic) stability that the (1,1) block of W, i.e. -X@A - A.conj().T@X,
    # is (pos. def.) pos. semidef.


def solve_Riccati(A_ph, B, C, D, solver):
    """
    Solve the continuous-time algebraic Riccati equation (CARE) for a given system.
    The function supports solving using two different solvers:
    `scipy` and `pymor`.

    Parameters:
    -----------
    A_ph : numpy.ndarray
        System matrix of the system.
    B : numpy.ndarray
        Input matrix of the system.
    C : numpy.ndarray
        Output matrix of the system.
    D : numpy.ndarray
        Feedthrough matrix of the system.
    solver : str
        Solver to use for solving the Riccati equation. Options are:
        - "scipy": Uses `scipy.linalg.solve_continuous_are`.
        - "pymor": Uses `pymor`'s LTIModel for solution.

    Returns:
    --------
    X : numpy.ndarray
        Solution to the Riccati equation, if the solver was successful.
    Qeye_system_exists : bool
        Flag indicating whether the Riccati equation was successfully solved.
    """

    if solver == "scipy":
        # Ricc(X) := -X@A - A.conj().T@X - (C.conj().T - X@B)@ inv(S) @ (C - B.conj().T@X) = 0
        A_Ricc = -A_ph
        B_Ricc = -B
        R_Ricc = D + D.conj().T  # called S in [BeattieMehrmannVanDooren18]
        S_Ricc = C.conj().T  # called C^H in [BeattieMehrmannVanDooren18]
        Q_Ricc = np.zeros_like(A_Ricc)
        try:
            logging.info(f"Trying to solve Riccati equation with scipy.")
            Qeye_system_exists = True
            X = linalg.solve_continuous_are(
                A_Ricc, B_Ricc, Q_Ricc, R_Ricc, e=None, s=S_Ricc, balanced=True
            )
            print(f"The Riccati equation was solved.")
        except np.linalg.LinAlgError as LinAlgError:
            print(
                f'LinAlgError: "{LinAlgError}" has occured. Riccati equation could not be solved. Q=I system is not calculated.'
            )
            Qeye_system_exists = False
    elif solver == "pymor":
        lti_model = LTIModel.from_matrices(A_ph, B, C, D)
        try:
            logging.info(f"Trying to solve Riccati equation with pymor/slycot.")
            phlti_model, X = PHLTIModel.from_passive_LTIModel(lti_model)
            Qeye_system_exists = True
        except:
            print(f"Error when calculating Q identity transformation")
            Qeye_system_exists = False

    return X, Qeye_system_exists


def Q_to_I_transformation(Qeye_system_exists, A_ph, B, C, D, X):
    """
    Perform a transformation of the system matrices based on the solution of the Riccati equation.

    This function checks the properties of the solution `X` to the Riccati equation, verifies the
    Kalman-Yakubovich-Popov (KYP) inequality, and performs a transformation to obtain a pH representation
    in transformed coordinates. It returns the transformed system matrices and the transformation matrices.

    Parameters:
    -----------
    Qeye_system_exists : bool
        Flag indicating whether the Riccati equation was successfully solved.
    A_ph : numpy.ndarray
        Descriptor matrix of the system.
    B : numpy.ndarray
        Input matrix of the system.
    C : numpy.ndarray
        Output matrix of the system.
    D : numpy.ndarray
        Direct transmission matrix of the system.
    X : numpy.ndarray
        Solution to the Riccati equation.

    Returns:
    --------
    J_T : numpy.ndarray
        Transformed matrix representing the pH system in T-coordinates.
    R_T : numpy.ndarray
        Transformed matrix representing the pH system in T-coordinates.
    B_T_ph : numpy.ndarray
        Transformed input matrix in T-coordinates.
    C_T_ph : numpy.ndarray
        Transformed output matrix in T-coordinates.
    T : numpy.ndarray
        Transformation matrix used for the coordinate change.
    T_inv : numpy.ndarray
        Inverse of the transformation matrix.
    """
    if Qeye_system_exists:
        # matrix function W, KYP-LMI holds if W(X)>=0 (W(X) semidefinite)
        W = lambda X: np.block(
            [
                [-X @ A_ph - A_ph.conj().T @ X, C.conj().T - X @ B],
                [C - B.conj().T @ X, D + D.conj().T],
            ]
        )
        # check KYP inequality
        eig_vals_X, eig_vectors_X = linalg.eig(X)
        if (eig_vals_X <= 0).any():
            print(f"The solution X is NOT pos. def.")
        else:
            print(f"Nice! The solution X is pos. def.")
        eig_vals_W, eig_vectors_W = linalg.eig(W(X))
        # eps = 1e-10
        if (eig_vals_W < 0).any():
            print(
                f"WARNING! The KYP inequality is not satisfied. Minimum eigenvalue: {eig_vals_W.min()}"
            )
        else:
            print(f"Nice! The KYP inequality is satisfied.")

        # if X is a solution to the KYP,
        # the symmetric factorization of X, e.g. Hermitian square root or Cholesky factorization
        use_cholesky = True
        if use_cholesky:
            T = linalg.cholesky(X)  # Cholesky factorization (X needs to be pos. def.)
            # T = T.conj().T  # This is wrong!
        else:
            # use Hermitian square root
            T = linalg.sqrtm(X)

        # leads to a transformed state-space in T-coordinates
        T_inv = np.linalg.inv(T)
        A_T = T @ A_ph @ T_inv
        B_T = T @ B
        C_T = C @ T_inv

        # we obtain a pH representation in T-coordinates
        J_T = 1 / 2 * (A_T - A_T.conj().T)
        R_T = -1 / 2 * (A_T + A_T.conj().T)
        K_T = 1 / 2 * (C_T.conj().T - B_T)
        G_T = 1 / 2 * (C_T.conj().T + B_T)

        # check pos. semidef. property of R_T
        eig_vals_R_T, eig_vectors_R_T = linalg.eig(R_T)
        if (eig_vals_R_T < 0).any():
            print(
                f"WARNING! The matrix R does NOT satisfy the pH pos. semidef. property. Minimal eigenvalue {eig_vals_R_T.min()}"
            )
        else:
            print(f"Nice! The matrix R does satisfy the pH pos. semidef. property")

        # build ph system matrices
        A_T_ph = J_T - R_T
        B_T_ph = G_T - K_T
        C_T_ph = (G_T + K_T).conj().T

    else:
        # return matrices as zeros
        A_T_ph = np.zeros_like(A_ph)
        B_T_ph = np.zeros_like(B)
        C_T_ph = np.zeros_like(C)

    return J_T, R_T, B_T_ph, C_T_ph, T, T_inv


def reshape_states_to_features(X, X_dt=None):
    """
    Transforms the state array 'X' into a feature array 'x', which is required for identification.

    Parameters:
    -----------
    X : ndarray
        State array of size (n_sim, n_t, n_n, n_dn) or (n_sim, n_t, n_f), where:
        - n_sim: number of simulations
        - n_t: number of time steps
        - n_n: number of nodes (only if X has 4 dimensions)
        - n_dn: number of degrees of freedom per node (only if X has 4 dimensions)
        - n_f: number of features (only if X has 3 dimensions)

    X_dt : ndarray, optional
        Time derivatives of the state array with the same shape as X. Default is None.

    Returns:
    --------
    x : ndarray
        Feature array of size (n_s, n_f), where:
        - n_s: number of samples (n_sim * n_t)
        - n_f: number of features (n_n * n_dn) or n_f

    dx_dt : ndarray, optional
        Feature array of the time derivatives of size (n_s, n_f). Returned only if X_dt is not None.

    Notes:
    ------
    - If X has 4 dimensions, it is reshaped from (n_sim, n_t, n_n, n_dn) to (n_s, n_f).
    - If X has 3 dimensions, it is assumed to be already in the shape (n_sim, n_t, n_f).
    - If X_dt is provided, it is reshaped similarly and returned alongside x.
    """
    if X.ndim == 4:
        # default
        n_sim, n_t, n_n, n_dn = X.shape
        n_f = n_n * n_dn
    elif X.ndim == 3:
        n_sim, n_t, n_f = X.shape

    n_s = n_sim * n_t
    x = X.reshape(n_s, n_f)
    if X_dt is not None:
        dx_dt = X_dt.reshape(n_s, n_f)
        return x, dx_dt
    return x


def reshape_inputs_to_features(U):
    """
    Transforms the input array 'U' into a feature array 'u', which is required for identification.
    The function reshapes the input array from (n_sim, n_t, n_u) to (n_s, n_u).

    Parameters:
    -----------
    U : ndarray
        Input array of size (n_sim, n_t, n_u), where:
        - n_sim: number of simulations
        - n_t: number of time steps
        - n_u: number of input features

    Returns:
    --------
    u : ndarray
        Feature array of size (n_s, n_u), where:
        - n_s: number of samples (n_sim * n_t)
        - n_u: number of input features
    """
    assert U.ndim == 3
    n_sim, n_t, n_u = U.shape
    n_s = n_sim * n_t
    u = U.reshape(n_s, n_u)

    return u


def reshape_features_to_states(x, n_sim, n_t, x_dt=None, n_n=None, n_dn=None, n_f=None):
    """
    Transforms the feature array 'x' back into a state array 'X', either 3-dimensional or 4-dimensional.

    Parameters:
    -----------
    x : ndarray
        Feature array of size (n_s, n_f), where:
        - n_s: number of samples (n_sim * n_t)
        - n_f: number of features (n_n * n_dn) if n_n and n_dn are provided, otherwise it is n_f.

    n_sim : int
        Number of simulations.
    n_t : int
        Number of time steps.
    x_dt : ndarray, optional
        Feature array of the time derivatives with the same shape as 'x'. Default is None.
    n_n : int, optional
        Number of nodes. Required if n_dn is provided. Default is None.
    n_dn : int, optional
        Number of degrees of freedom per node. Required if n_n is provided. Default is None.
    n_f : int, optional
        Number of features. Default is None.

    Returns:
    --------
    X : ndarray
        State array of size (n_sim, n_t, n_n, n_dn) if n_n and n_dn are provided, otherwise (n_sim, n_t, n_f).
    X_dt : ndarray, optional
        State array of the time derivatives with the same shape as 'X'. Returned only if x_dt is not None.

    Raises:
    -------
    ValueError
        If both (n_n and n_dn) and n_f are provided.

    Notes:
    ------
    - Either n_f should be provided or both n_n and n_dn should be provided, but not both.
    - If n_n and n_dn are provided, the function reshapes the feature array to (n_sim, n_t, n_n, n_dn).
    - If n_f is provided, the function reshapes the feature array to (n_sim, n_t, n_f).
    - If x_dt is provided, it is reshaped similarly and returned alongside X.
    """
    if (n_n != None or n_dn != None) and n_f != None:
        raise ValueError("Choose either n_n and n_dn or only n_f")
    # return ndim=4 states
    if n_n != None:
        X = x.reshape(n_sim, n_t, n_n, n_dn)
        if x_dt is not None:
            X_dt = x_dt.reshape(n_sim, n_t, n_n, n_dn)
            return X, X_dt
    # return ndim=3 states
    elif n_f != None:
        X = x.reshape(n_sim, n_t, n_f)
        if x_dt is not None:
            X_dt = x_dt.reshape(n_sim, n_t, n_f)
            return X, X_dt
    return X
