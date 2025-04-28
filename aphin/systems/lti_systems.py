"""
Define and simulate dynamical systems
e.g. LTI systems (LTISystem), port-Hamiltonian systems (PHSystem), ...
"""

import numpy as np
from scipy import signal
from aphin.utils.integrators import implicit_midpoint


class LTISystem:
    """
    General linear, time invariant ODE system (LTI system) with constant system matrices
    """

    def __init__(self, A, B=None):
        """
        Initializes a LTISystem class instance with system matrix A.

        The system is defined by the differential equation:
        x'(t) = A * x(t) + B * u(t), x(0) = x_init

        Parameters
        ----------
        A : array-like, shape (n, n)
            System matrix.
        B : array-like, shape (n, n_u), optional
            Input matrix. If not provided, it defaults to a zero matrix of shape (n, 1).
        """

        super(LTISystem, self).__init__()
        self.A = A
        if B is None:
            self.B = np.zeros((self.A.shape[0], 1))
        else:
            if B.ndim == 3:
                self.B = np.squeeze(B, axis=2)
            else:
                self.B = B
        assert self.A.shape[0] == self.A.shape[1]
        assert self.A.shape[0] == self.B.shape[0]
        self.n = self.A.shape[0]
        self.n_u = self.B.shape[1]
        self.C = np.eye(self.n)
        self.E = np.eye(self.n)
        self.E_inv_A = self.A
        self.E_inv_B = self.B
        self.sys = signal.StateSpace(self.A, self.B, self.C)

    def solve(self, t, z_init, u=None, integrator_type="IMR", decomp_option="lu"):
        """
        Solve the system for given times, initial conditions, and inputs.

        Parameters
        ----------
        t : array-like, shape (n_t,)
            Array with time steps for the solution.
        z_init : array-like, shape (n,) or (n, n_sim)
            Initial conditions of the system. If the shape is (n,), it represents a single simulation;
            if the shape is (n, n_sim), it represents multiple simulations.
        u : array-like, shape (n_t, n_u) or (n_sim, n_t, n_u), optional
            Input signal for the system. If not provided, it defaults to zeros.
            The shape should match the dimensions of the input matrix B.
        integrator_type : {'IMR', 'lsim'}, optional
            The integration method to use. 'IMR' stands for implicit midpoint rule, and 'lsim' uses the matrix exponential.
            Default is 'IMR'.
        decomp_option : {'lu', 'linalg_solve'}, optional
            The decomposition option for the IMR integrator. 'lu' uses LU decomposition,
            and 'linalg_solve' solves without decomposition. Default is 'lu'.

        Returns
        -------
        z_sol : ndarray, shape (n_sim, n_t, n_f)
            The solution of the ODE system. If multiple simulations are performed,
            the result is a 3D array where each slice corresponds to a simulation.
        """

        assert isinstance(integrator_type, str)

        n_t = len(t)
        if z_init.ndim == 1:
            z_init = np.expand_dims(z_init, axis=1)
        n_f, n_sim = z_init.shape
        n_u = self.B.shape[1]
        if u is None:
            u_ = np.zeros((n_sim, n_t, n_u))
        elif u.ndim == 3:
            u_ = u.copy()
        else:
            raise ValueError("Shape of u does not fit")
        assert self.A.shape[0] == n_f
        z_sol = np.zeros((n_sim, n_t, n_f))
        for i in range(n_sim):
            u_i = u_[i, :, :]
            if integrator_type.lower() == "imr":
                z_out, _ = implicit_midpoint(
                    self.E,
                    self.A,
                    t.ravel(),
                    z_init[:, i],
                    B=self.B,
                    u=u_i,
                    decomp_option=decomp_option,
                )
            elif integrator_type.lower() == "lsim":
                _, _, z_out = signal.lsim(
                    self.sys, u_i, t.ravel(), X0=z_init[:, i], interp=True
                )
            else:
                raise ValueError(
                    f"Input value of solve_method={integrator_type} is unknown"
                )
            z_sol[i, :, :] = z_out
        return z_sol

    def solve_dt(self, t, z_init, u=None, integrator_type="IMR", decomp_option="lu"):
        """
        Solves the system of ordinary differential equations (ODEs) for given time steps, initial conditions, and inputs.
        Also computes the exact time derivative of the solution.

        Parameters:
        -----------
        t : ndarray
            Array of time steps with shape (n_t,).

        z_init : ndarray
            Initial conditions with shape (n_f, n_sim) or (n_f,).

        U : ndarray, optional
            Input signal with shape (n_t, n_u) or (n_sim, n_t, n_u). Default is None.
            If None, a zero input is used.

        integrator_type : str, optional
            Type of integrator to use. Choices are:
            - 'IMR' : Implicit Midpoint Rule
            - 'LSIM' : Matrix Exponential (Default is 'IMR').

        decomp_option : str, optional
            Decomposition option for the Implicit Midpoint Rule. Choices are:
            - 'lu' : LU decomposition
            - 'linalg_solve' : Solve without decomposition (Default is 'lu').

        Returns:
        --------
        x : ndarray
            Solution of the ODE system with shape (n_sim, n_t, n_f).

        dx_dt : ndarray
            Time derivative of the solution with shape (n_sim, n_t, n_f).
        """
        n_t = len(t)
        if z_init.ndim == 1:
            z_init = np.expand_dims(z_init, axis=1)
        n_f, n_sim = z_init.shape
        assert self.A.shape[0] == n_f
        n_u = self.B.shape[1]
        if u is None:
            u_ = np.zeros((n_sim, n_t, n_u))
        elif u.ndim == 2:
            u_ = u.copy().reshape(n_sim, n_t, n_u)
        elif u.ndim == 3:
            u_ = u.copy()
        else:
            raise ValueError("Shape of u does not fit")
        u_midpoint = (u_[:, 1:] + u_[:, :-1]) / 2
        u_[:, :-1] = u_midpoint
        z = self.solve(
            t, z_init, u_, integrator_type=integrator_type, decomp_option=decomp_option
        )
        dz_dt = np.empty((n_sim, n_t, n_f))
        for i in range(n_sim):
            dz_dt_i = (
                self.A @ z[i, :, :].transpose() + self.B @ u_[i, :, :].transpose()
            )  # dx_dt_i of size (n_f,n_t)
            dz_dt[i, :, :] = dz_dt_i.transpose()  # of size (n_sim,n_t,n_f)
        return z, dz_dt

    @property
    def stable(self, eps=1e-4):
        """
        Check if the system is stable.

        This property calculates the eigenvalues of the matrix `E_inv@A` and checks if the real part
        of the largest eigenvalue is less than a negative threshold, indicating stability.

        Parameters
        ----------
        eps : float, optional
            A small positive threshold to determine stability. Default is 1e-4.

        Returns
        -------
        bool
            True if the system is stable, False otherwise.
        """
        return np.linalg.eigvals(self.E_inv_A).max().real < -eps

    def is_regular(self, M):
        """
        Check if a matrix M is regular.

        A matrix is considered regular if it has full rank and a condition number below a specified threshold.

        Parameters
        ----------
        M : array-like, shape (n, n)
            The matrix to be checked for regularity.

        Returns
        -------
        bool
            True if the matrix M is regular, False otherwise.
        """
        return np.linalg.matrix_rank(M) == self.n and np.linalg.cond(M) < 1e10

    @staticmethod
    def is_sym(M):
        """
        Check if a matrix M is symmetric.

        This method checks whether the matrix M is equal to its transpose.

        Parameters
        ----------
        M : array-like, shape (n, n)
            The matrix to be checked for symmetry.

        Returns
        -------
        bool
            True if the matrix M is symmetric, False otherwise.
        """
        return np.allclose(M, M.T)

    @staticmethod
    def is_skew_sym(M):
        """
        Check if a matrix M is skew-symmetric.

        This method checks whether the matrix M is equal to the negative of its transpose.

        Parameters
        ----------
        M : array-like, shape (n, n)
            The matrix to be checked for skew-symmetry.

        Returns
        -------
        bool
            True if the matrix M is skew-symmetric, False otherwise.
        """
        return np.allclose(M, -M.T)

    @staticmethod
    def is_pos_def(M):
        """
        Check if a matrix M is positive semi-definite.

        This method checks whether all eigenvalues of the matrix M are non-negative. The matrix is considered
        positive semi-definite if the smallest eigenvalue is greater than or equal to a small threshold
        determined by the machine precision.

        Parameters
        ----------
        M : array-like, shape (n, n)
            The matrix to be checked for positive semi-definiteness.

        Returns
        -------
        bool
            True if the matrix M is positive semi-definite, False otherwise.
        """
        return np.linalg.eigvals(M).min() >= -np.finfo(M.dtype).eps

    @staticmethod
    def quad(xl, M, xr=None):
        """
        Calculate the quadratic form \( xl^T @ M @ xr \).

        This method computes the quadratic form given by the expression:
        xl^T @ M @ xr, where xl and xr are vectors and M is a matrix.

        Parameters
        ----------
        xl : array-like, shape (n,)
            Vector on the left side of the matrix M.
        M : array-like, shape (n, n)
            Matrix representing the quadratic form.
        xr : array-like, shape (n,), optional
            Vector on the right side of the matrix M. If not provided, xr is assumed to be equal to xl.

        Returns
        -------
        float
            The result of the quadratic form.
        """
        if xr is None:
            xr = xl
        return np.einsum("ti,ij,tj->t", xl, M, xr)

    def get_system_matrix(self):
        """
        Retrieve the system matrices A, B, and C.

        This method returns the matrices that define the linear time-invariant (LTI) system.

        Returns
        -------
        tuple of array-like
            A tuple containing the system matrices:
            - A : array-like, shape (n, n)
                The state matrix.
            - B : array-like, shape (n, n_u)
                The input matrix.
            - C : array-like, shape (n, n)
                The output matrix.
        """
        return self.A, self.B, self.C


class DescrLTISystem(LTISystem):
    """
    Linear time-invariant (LTI) system with descriptor formulation.

    This class represents a linear time-invariant ODE system in a descriptor formulation, where
    the system is described by the equation:
    E * x'(t) = A * x(t) + B * u(t), with initial condition x(0) = x_init.

    The descriptor matrix E allows for more general system representations, including
    differential-algebraic equations (DAEs) if E is singular.
    """

    def __init__(self, A, B=None, E=None):
        """
        Initialize a descriptor linear time-invariant (LTI) system.

        Parameters
        ----------
        A : array-like, shape (n, n)
            System matrix.
        B : array-like, shape (n, n_u), optional
            Input matrix. If not provided, defaults to a zero matrix of appropriate shape.
        E : array-like, shape (n, n), optional
            Descriptor matrix. If not provided, defaults to the identity matrix of appropriate shape.
        """
        super(DescrLTISystem, self).__init__(A, B)
        if E is None:
            self.E = np.eye(self.A.shape[0])
        else:
            self.E = E
        self.E_inv_A = np.linalg.solve(self.E, self.A)
        self.E_inv_B = np.linalg.solve(self.E, self.B)
        self.sys = signal.StateSpace(self.E_inv_A, self.E_inv_B, self.C)

    def solve(self, t, z_init, u=None, integrator_type="IMR", decomp_option="lu"):
        """
        Solve the descriptor linear time-invariant (LTI) system for given times, initial conditions, and inputs.

        This method integrates the system's differential-algebraic equations over the specified time steps.

        Parameters
        ----------
        t : array-like, shape (n_t,)
            Array of time steps for which the solution is computed.
        z_init : array-like, shape (n,) or (n, n_s)
            Initial conditions of the system. If multiple simulations are run, this should have shape (n, n_s).
        u : array-like, shape (n_t, n_u) or (n_t, n_u, n_s), optional
            Input signal for the system. If not provided, defaults to zero input. If multiple simulations are run,
            the shape should be (n_t, n_u, n_s).
        integrator_type : {'IMR'}, optional
            The type of integrator to use. Only 'IMR' (Implicit Midpoint Rule) is supported for descriptor systems.
        decomp_option : {'lu', 'linalg_solve'}, optional
            Option for decomposition in the IMR integrator. Options are 'lu' for LU decomposition or 'linalg_solve'
            for solving without decomposition.

        Returns
        -------
        ndarray, shape (n_t, n, n_s)
            The solution of the ODE system at each time step.
        """
        if integrator_type == "lsim":
            raise ValueError(
                f"Integrator type {integrator_type} is not supported for descriptor systems."
            )
        return super(DescrLTISystem, self).solve(
            t,
            z_init,
            u=u,
            integrator_type="IMR",
            decomp_option=decomp_option,
        )
