import numpy as np
import scipy
import logging
from timeit import default_timer as timer

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def implicit_midpoint(E, A, t, z_init, B=None, u=None, decomp_option="lu"):
    """
    Calculate time integration of a linear ODE system using the implicit midpoint rule.

    This function solves the ODE system defined by:
        E * dz_dt = A * z + B * u

    Parameters:
    -----------
    E : numpy.ndarray
        Descriptor matrix of the ODE system.
    A : numpy.ndarray
        System matrix of the ODE system.
    t : numpy.ndarray
        Time vector with equidistant time steps.
    z_init : numpy.ndarray
        Initial state vector.
    B : numpy.ndarray, optional
        Input matrix. Defaults to None, which sets B to zero.
    u : numpy.ndarray, optional
        Input function at time midpoints. Defaults to None, which sets u to zero.
    decomp_option : str, optional
        Option for matrix decomposition or solving. Choices are 'lu' for LU decomposition or
        'linalg_solve' for solving without decomposition. Defaults to 'lu'.

    Returns:
    --------
    z : numpy.ndarray
        Array of state vectors at each time step.
    t : numpy.ndarray
        Array of time points.

    Theory:
    --------
    we got a pH-system E*Dx = (J-R)*Q*x + B*u
    we define A:=(J-R)*Q and the RHS as f(t,x)
    use the differential slope equation at midpoint
    (x(t+h)-x(t))/h=Dx(t+h/2)=E^-1 * f(t+h/2,x(t+h/2))
    since x(t+h/2) is unknown we use the approximation
    x(t+h/2) = 1/2*(x(t)+x(t+h))
    insert the linear system into the differential equation leads to
    x(t+h) = x(t) + h * E^-1 *(1/2*A*(x(t)+x(t+h))+ B*u(t+h/2))
    reformulate the equation to
    (E-h/2*A)x(t+h) = (E+h/2*A)*x(t) + h*B*u(t+h/2)
    solve the linear equation system, e.g. via LU-decomposition

    The linear system is solved for x(t + h) using the specified decomposition method.

    Examples:
    ---------
    >>> E = np.array([[1, 0], [0, 1]])
    >>> A = np.array([[0, -1], [1, 0]])
    >>> t = np.linspace(0, 10, 100)
    >>> z_init = np.array([1, 0])
    >>> z, t = implicit_midpoint(E, A, t, z_init)
    """

    # number of time samples
    n_t = len(t)
    step_size = t[1] - t[0]
    # assert np.allclose(t[1:] - t[:-1], [step_size] * (n_t - 1))  # constant time steps
    tmid = np.zeros(n_t)
    # initialize state array
    n_f = len(z_init)
    z = np.zeros((n_t, n_f))
    z[0, :] = z_init

    # default input values
    if B is None:
        n_u = 1
        B = np.zeros((n_f, n_u))
        u = np.zeros((n_t, n_u))
    else:
        logging.info(
            "Use predefined input function u. Make sure that inputs are given at mid timepoints"
        )

    # time different decomposition methods
    start_time_solver = timer()

    if decomp_option == "lu":
        # LU decomposition of lhs
        lu, piv = scipy.linalg.lu_factor((E - (step_size / 2) * A))

    # loop over time samples
    for i_t in range(n_t - 1):
        tmid[i_t] = t[0] + ((i_t + 1) - 0.5) * step_size
        if decomp_option == "lu":
            z[i_t + 1, :] = scipy.linalg.lu_solve(
                (lu, piv),
                (E + (step_size / 2) * A) @ z[i_t, :] + step_size * B @ u[i_t, :],
            )
        elif decomp_option == "linalg_solve":
            z[i_t + 1, :] = np.linalg.solve(
                (E - (step_size / 2) * A),
                (E + (step_size / 2) * A) @ z[i_t, :] + step_size * B @ u[i_t, :],
            )
        else:
            raise ValueError(f"Decomposition option {decomp_option} is unknown.")

    end_time_solver = timer()
    solver_time = end_time_solver - start_time_solver
    logging.info(f"Calculation time of ODE solver has been {solver_time} s.\n")

    return z, t[:, np.newaxis]
