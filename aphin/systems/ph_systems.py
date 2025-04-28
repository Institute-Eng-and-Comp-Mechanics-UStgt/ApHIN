"""
Define and simulate dynamical systems
e.g. LTI systems (LTISystem), port-Hamiltonian systems (PHSystem), ...
"""

import warnings
import numpy as np
import logging
from aphin.systems import LTISystem, DescrLTISystem
from aphin.utils.transformations import transform_pH_to_Q_identity


class CheckPHProperties:
    """
    A class for checking properties of matrices related to port-Hamiltonian (pH) systems.
    """

    def check_pH_properties(self, J, R, Q, E=None, rtol=1e-05, atol=1e-08):
        """
        Check if the matrices J, R, and Q (with optional descriptor E) satisfy port-Hamiltonian system properties.

        This method verifies the following properties:
        - J is skew-symmetric.
        - R is symmetric positive definite.
        - transpose(E)@Q is symmetric positive definite

        Parameters
        ----------
        J : array-like, shape (n, n)
            Skew-symmetric matrix related to the port-Hamiltonian system.
        R : array-like, shape (n, n)
            Symmetric positive definite matrix.
        Q : array-like, shape (n, n), optional
            Matrix to be checked for positive definiteness.
        E : array-like, shape (n, n), optional
            Descriptor matrix. If not provided, defaults to the identity matrix.
        rtol : float, optional
            Relative tolerance for numerical comparison (default is 1e-05).
        atol : float, optional
            Absolute tolerance for numerical comparison (default is 1e-08).

        Returns
        -------
        bool
            True if all properties are satisfied, False otherwise.
        """

        J_check = np.allclose(J, -J.T, rtol=rtol, atol=atol)
        R_check = self.check_spd(R.copy())

        if E is None:
            E = np.eye(J.shape[0])
        if Q is not None:
            Q_check = self.check_spd(E.T @ Q)
        else:
            Q_check = True

        is_ph = J_check and R_check and Q_check
        if not is_ph:
            logging.warning(f"The system does NOT satisfy the pH properties.")

        return is_ph

    def check_spd(self, A, rtol=1e-06, atol=1e-08):
        """
        Check if a matrix is symmetric positive definite.

        This method verifies if the matrix `A` is symmetric and positive definite. It does so by attempting
        a Cholesky decomposition. If the decomposition fails due to the matrix not being positive definite,
        a small regularization term is added to the matrix to try and make it positive definite.

        Parameters
        ----------
        A : array-like, shape (n, n)
            The matrix to be checked for symmetric positive definiteness.
        rtol : float, optional
            Relative tolerance for checking symmetry of the matrix (default is 1e-06).
        atol : float, optional
            Absolute tolerance for checking symmetry of the matrix (default is 1e-08).

        Returns
        -------
        bool
            True if the matrix is symmetric positive definite, False otherwise.

        References
        ----------
        Adapted from [MorandinNicodemusUnger22].
        """
        if np.allclose(A, A.T, rtol=rtol, atol=atol):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                A += 1e-8 * np.eye(A.shape[0])
                try:
                    np.linalg.cholesky(A)
                    return True
                except np.linalg.LinAlgError:
                    return False
        else:
            return False


class PHSystem(LTISystem, CheckPHProperties):
    """
    General linear, port-Hamiltonian (pH) system with constant system matrices.

    This class represents a linear port-Hamiltonian system, defined by the differential equation:

        x'(t) = (J - R) * x(t) + B * u(t), x(0) = x_init

    where:
    - `J` is a skew-symmetric matrix (J = -J.T).
    - `R` is a symmetric positive semi-definite matrix (R = R.T >= 0).
    """

    def __init__(self, J_ph, R_ph, B=None, Q_ph=None):
        """
        Initializes the port-Hamiltonian system with the given matrices.

        Parameters
        ----------
        J_ph : array-like, shape (n, n)
            The port-Hamiltonian J matrix, which should be skew-symmetric.
        R_ph : array-like, shape (n, n)
            The port-Hamiltonian R matrix, which should be symmetric positive semi-definite.
        B : array-like, shape (n, n_u), optional
            The input matrix.
        Q_ph : array-like, shape (n, n), optional
            The Q matrix associated with the port-Hamiltonian system. If not provided, the identity matrix is used.
        """
        if J_ph.shape[0] == 1:
            J_ph = np.squeeze(J_ph, axis=0)
        self.J_ph = J_ph
        if R_ph.shape[0] == 1:
            R_ph = np.squeeze(R_ph, axis=0)
        self.R_ph = R_ph
        if Q_ph is None:
            self.Q_ph = np.eye(*self.J_ph.shape)
            self.Q_is_identity = True
        else:
            if Q_ph.shape[0] == 1:
                Q_ph = np.squeeze(Q_ph, axis=0)
            self.Q_ph = Q_ph
            self.Q_is_identity = False
        assert self.J_ph.shape[0] == self.J_ph.shape[1]
        assert self.R_ph.shape[0] == self.R_ph.shape[1]
        assert self.J_ph.shape[0] == self.R_ph.shape[0]
        if self.J_ph.ndim == 3 or self.R_ph.ndim == 3 or self.Q_ph.ndim == 3:
            raise ValueError("Insert pH square matrix of size (r,r)")
        self.check_pH_properties(self.J_ph, self.R_ph, self.Q_ph)
        if B is not None:
            if B.ndim == 3:
                B = np.squeeze(B, axis=0)
            self.B_ph = B
            self.C_ph = self.B_ph.T @ self.Q_ph
        else:
            self.B_ph = B
            self.C_ph = None
        A = (self.J_ph - self.R_ph) @ self.Q_ph
        super(PHSystem, self).__init__(A, self.B_ph)

    def H(self, x):
        """
        Calculate the Hamiltonian for a time series of states.

        Computes the Hamiltonian for a time series of states `x`, given by the quadratic form:

            H(x) = 0.5 * x.T @ Q_ph @ x

        Parameters
        ----------
        x : array-like, shape (n_t, n)
            Time series of states where `n_t` is the number of time steps and `n` is the dimension of the state vector.

        Returns
        -------
        ndarray, shape (n_t,)
        The Hamiltonian values corresponding to each time step.
        """
        return 0.5 * self.quad(x, self.Q_ph, x)

    def transform_pH_to_Q_identity(self, solver="Q", seed=1):
        """
        Transform the current port-Hamiltonian (pH) system to a Q = I (identity matrix) form, if possible.

        This method attempts to transform the port-Hamiltonian system described by the matrices
        `J_ph`, `R_ph`, `Q_ph`, `B_ph`, and `C_ph` to a new system where `Q` is replaced by the identity matrix.
        If the transformation is successful, it returns a new `PHSystem` instance with the transformed matrices.
        If the transformation fails, it logs a message and returns the current instance.

        Parameters
        ----------
        solver : {'scipy', 'pymor', 'Q'}, optional
            Method for solving the transformation:
            - 'scipy' or 'pymor': Use the Riccati equation solver.
            - 'Q': Use the provided Q matrix directly.
        seed : int, optional
            Random seed for generating a dummy input matrix when solving the Riccati equation.

        Returns
        -------
        PHSystem or self
            If the transformation to Q = I is successful, returns a new `PHSystem` instance with the transformed matrices.
            Otherwise, returns the current instance.
        T : ndarray, shape (n, n)
            The transformation matrix, only returned if the transformation is successful.
        T_inv : ndarray, shape (n, n)
            The inverse of the transformation matrix, only returned if the transformation is successful.
        """

        J_T, R_T, B_T_ph, C_T_ph, T, T_inv, input_used, Qeye_system_exists = (
            transform_pH_to_Q_identity(
                self.J_ph,
                self.R_ph,
                self.Q_ph,
                self.B_ph,
                self.C_ph,
                solver=solver,
                seed=seed,
            )
        )

        if Qeye_system_exists:
            # change the pH system to Q identity form
            return PHSystem(J_T, R_T, B_T_ph), T, T_inv
        else:
            logging.info(
                f"The Q identity form could not be found. PHSystem instance is not changed."
            )
            return self

    def get_system_matrix(self):
        return self.J_ph, self.R_ph, self.B_ph, self.Q_ph


class DescrPHSystem(DescrLTISystem, CheckPHProperties):
    """
    Linear port-Hamiltonian (pH) system in descriptor form with constant system matrices.

    This class represents a linear port-Hamiltonian system described by the following differential equation:
    E * x'(t) = (J - R) * x(t) + B * u(t), with x(0) = x_init, where:
    - J is skew-symmetric: J = -J.T
    - R is symmetric positive semi-definite: R = R.T >= 0
    - E is a descriptor matrix
    """

    def __init__(self, J_ph, R_ph, E_ph, B=None, Q_ph=None):
        """
         Parameters
        ----------
        J_ph : ndarray, shape (n, n)
            Port-Hamiltonian J matrix. Must be skew-symmetric.
        R_ph : ndarray, shape (n, n)
            Port-Hamiltonian R matrix. Must be symmetric positive semi-definite.
        E_ph : ndarray, shape (n, n)
            Descriptor matrix.
        B : ndarray, shape (n, n_u), optional
            Input matrix. Defaults to None if not provided.
        Q_ph : ndarray, shape (n, n), optional
            Port-Hamiltonian Q matrix. Defaults to the identity matrix if not provided.
        """
        self.J_ph = np.squeeze(J_ph)
        self.R_ph = np.squeeze(R_ph)
        self.E_ph = np.squeeze(E_ph)
        if Q_ph is None:
            self.Q_ph = np.eye(*self.J_ph.shape)
        else:
            self.Q_ph = np.squeeze(Q_ph)
        if B is not None:
            self.B_ph = np.squeeze(B)
            self.C_ph = self.B_ph.T @ self.Q_ph
        else:
            self.B_ph = B
            self.C_ph = None
        assert self.J_ph.shape[0] == self.J_ph.shape[1]
        assert self.R_ph.shape[0] == self.R_ph.shape[1]
        assert self.J_ph.shape[0] == self.R_ph.shape[0]
        if (
            self.J_ph.ndim == 3
            or self.R_ph.ndim == 3
            or self.Q_ph.ndim == 3
            or self.E_ph.ndim == 3
        ):
            raise ValueError("Insert pH square matrix of size (r,r)")

        # assert self.check_pH_properties(self.J_ph, self.R_ph, self.Q_ph, self.E)
        A = (self.J_ph - self.R_ph) @ self.Q_ph
        super(DescrPHSystem, self).__init__(A=A, B=self.B_ph, E=self.E_ph)

    def solve(self, t, z_init, u=None, integrator_type="IMR", decomp_option="lu"):
        """
        Solve the descriptor port-Hamiltonian system using the specified integration method.

        Computes the solution to the differential equation system E * x'(t) = (J - R) * x(t) + B * u(t)
        over the given time steps with the provided initial conditions and inputs.

        Parameters
        ----------
        t : ndarray, shape (n_t,)
            Array of time steps for the solution.
        z_init : ndarray, shape (n,) or (n, n_s)
            Initial conditions for the system state.
        u : ndarray, shape (n_t, n_u) or (n_t, n_u, n_s), optional
            Input signal to the system. If not provided, assumes zero input.
        integrator_type : str, optional
            Integration method to use. Default is "IMR" (Implicit Midpoint Rule).
        decomp_option : str, optional
            Decomposition method for the integrator. Default is "lu". Other options may be available depending on the integrator.

        Returns
        -------
        x : ndarray, shape (n_t, n, n_s)
            Solution of the ODE system at each time step.
        """
        return super(DescrPHSystem, self).solve(
            t,
            z_init,
            u=u,
            integrator_type=integrator_type,
            decomp_option=decomp_option,
        )

    def H(self, x):
        """
        Calculate the Hamiltonian for a time series of states.

        Computes the Hamiltonian function H(x) = 0.5 * x.T @ (E.T @ Q_ph) @ x for each time step in the time series x.

        Parameters
        ----------
        x : ndarray, shape (n_t, n)
            Time series of states where n_t is the number of time steps and n is the dimension of the state vector.

        Returns
        -------
        ndarray, shape (n_t,)
            The Hamiltonian values computed for each time step in the time series x.
        """
        return 0.5 * self.quad(x, self.E.T @ self.Q_ph, x)
