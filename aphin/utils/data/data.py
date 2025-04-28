"""
Encapsulate data loading and data generation
"""

from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import os
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from tqdm import tqdm

# own package
from aphin.utils.transformations import (
    reshape_states_to_features,
    reshape_inputs_to_features,
    reshape_features_to_states,
)
from aphin.systems import PHSystem, DescrPHSystem
from aphin.identification import PHIN, APHIN
from aphin.layers import (
    PHLayer,
    PHQLayer,
    LTILayer,
    DescriptorPHLayer,
    DescriptorPHQLayer,
)


class Data(ABC):
    """
    Container class for datasets in a linear dynamical system framework.

    This class is designed to store and manage various datasets required for system analysis and identification, including states, time derivatives, inputs, parameters, and port-Hamiltonian matrices.
    """

    def __init__(
        self,
        t,
        X,
        X_dt=None,
        U=None,
        Mu=None,
        J=None,
        R=None,
        Q=None,
        B=None,
        Mu_input=None,
    ):
        """
        Initializes the Data container with multiple datasets (train and test) including states, inputs, and parameters.

        Parameters:
        -----------
        t : ndarray
            Array of time steps. (n_t,1)
        X : ndarray
            States array with shape (n_sim, n_t, n_n, n_dn).
        X_dt : ndarray
            Time derivatives of the states with the same shape as X. Default is None.
        U : ndarray, optional
            Input array with shape (n_sim, n_t, n_u). Default is None.
        Mu : ndarray, optional
            Parameters array with shape (n_sim, n_mu). Default is None.
        J : ndarray, optional
            pH interconnection matrix with shape (n_n * n_dn, n_n * n_dn). Default is None.
        R : ndarray, optional
            pH dissipation matrix with shape (n_n * n_dn, n_n * n_dn). Default is None.
        Q : ndarray, optional
            pH energy matrix with shape (n_n * n_dn, n_n * n_dn). Default is None.
        B : ndarray, optional
            pH input matrix with shape (n_n * n_dn, n_u). Default is None.
        Mu_input : ndarray, optional
            Input parameters used to generate U, if applicable. Default is None.

        Notes:
        ------
        - If X_dt is not provided, it is computed as the gradient of X with respect to t.
        - The matrices J, R, Q, and B, if provided, are broadcasted to have the same number of simulations as X.
        - This initializer sets up the structure for data containers, which can then be used for further processing and analysis in derived classes.
        """

        if t is None:
            raise ValueError("time steps t are not given")
        if X is None:
            raise ValueError("states X are not given")
        if X.ndim != 4:
            raise ValueError(
                "states X need to be given in the format (n_sim, n_t, n_n, n_dn)"
            )
        logging.info(
            "Reading states X. Ensure that states are of size (n_sim, n_t, n_n, n_dn)"
        )
        self.n_sim, self.n_t, self.n_n, self.n_dn = X.shape

        if self.n_t != len(t):
            raise ValueError("number of time steps in t and in states X does not match")
        if t.ndim == 1:
            t = t[:, np.newaxis]
        self.t = t

        self.X = X
        if X_dt is None:
            # Compute time derivatives
            self.X_dt = np.gradient(self.X, t.ravel(), axis=1)
        else:
            self.X_dt = X_dt

        if U is None:
            self.U = None
            self.n_u = 0
        else:
            self.U = U
            if self.U.ndim != 3:
                raise ValueError(
                    "input needs to be given in the format (n_sim,n_t,n_u)"
                )
            if self.n_t != self.U.shape[1]:
                raise ValueError("number of time steps in t and in u does not match")
            if self.n_sim != self.U.shape[0]:
                raise ValueError(
                    "number of simulations in states X and in U does not match"
                )
            self.n_u = self.U.shape[2]

        if Mu is None:
            self.Mu = None
            self.n_mu = 0
        else:
            self.Mu = Mu
            self.n_mu = Mu.shape[1]
            if self.n_sim != Mu.shape[0]:
                raise ValueError(
                    "number of scenarios in mu and in states X does not match"
                )

        self.add_ph_matrices(J, R, B=B, Q=Q)

        self.Mu_input = Mu_input

        # Initialize x, dx_dt and x_init. Subclasses should override these values
        self.x = None
        self.dx_dt = None
        self.u = None
        self.mu = None
        self.x_init = None
        # scaling
        self.scaling_values = None
        self.u_train_bounds = None
        self.mu_train_bounds = None
        self.is_scaled = False

    @property
    def data(self):
        """
        Returns the stored datasets for time, states, state derivatives, inputs, and parameters.

        Returns:
        --------
        tuple
            A tuple containing:
            - t: Array of time steps.
            - x: States array.
            - dx_dt: Time derivatives of the states.
            - u: Input array.
            - mu: Parameters array.
        """
        return self.t, self.x, self.dx_dt, self.u, self.mu

    @property
    def ph_matrices(self):
        """
        Returns the port-Hamiltonian matrices used in the data container.

        Returns:
        --------
        tuple
            A tuple containing:
            - J: Port-Hamiltonian interconnection matrix.
            - R: Port-Hamiltonian dissipation matrix.
            - Q: Port-Hamiltonian energy matrix.
            - B: Port-Hamiltonian input matrix.
        """
        return self.J, self.R, self.Q, self.B

    @property
    def Data(self):
        """
        Retrieve the state and derivative data from the container.

        Returns:
        --------
        tuple
            A tuple containing:
            - X: States array with shape (n_sim, n_t, n_n, n_dn).
            - X_dt: Time derivatives of the states with the same shape as X.
            - U: Input array with shape (n_sim, n_t, n_u), if available.
            - Mu: Parameters array with shape (n_sim, n_mu), if available.
        """
        return self.X, self.X_dt, self.U, self.Mu

    @property
    def shape(self):
        """
        Return the shape of the dataset.

        Returns:
        --------
        tuple
            A tuple containing:
            - n_sim: number of simulations
            - n_t: number of time steps
            - n_n: number of nodes
            - n_dn: number of degrees of freedom per node
            - n_u: number of inputs
            - n_mu: number of parameters
        """
        return self.n_sim, self.n_t, self.n_n, self.n_dn, self.n_u, self.n_mu

    def permute_matrices(self, permutation_idx: list[int] | slice):
        """
        Permute the dimensions of the port-Hamiltonian (pH) matrices (J, R, Q, B)
        according to a given permutation index.

        This method modifies the order of states and corresponding matrix entries
        based on the supplied permutation index. It supports both list-based index
        permutations and slicing. Permutations are applied along the last two axes
        for square matrices and along the second axis for the input matrix B.

        Parameters
        ----------
        permutation_idx : list[int] or slice
            Either a list specifying the new order of indices or a slice object to select
            a contiguous range of indices.

        Raises
        ------
        AssertionError
            If the length of the permutation list does not match the expected matrix dimension.

        ValueError
            If the provided permutation index is neither a list nor a slice.
        """
        if isinstance(permutation_idx, list):
            assert len(permutation_idx) == self.J.shape[1]
            self.J = self.J[:, permutation_idx, :][:, :, permutation_idx]
            self.R = self.R[:, permutation_idx, :][:, :, permutation_idx]
            self.Q = self.Q[:, permutation_idx, :][:, :, permutation_idx]
            self.B = self.B[:, permutation_idx, :]
        elif isinstance(permutation_idx, slice):
            self.J = self.J[:, permutation_idx, permutation_idx]
            self.R = self.R[:, permutation_idx, permutation_idx]
            self.Q = self.Q[:, permutation_idx, permutation_idx]
            self.B = self.B[:, permutation_idx, permutation_idx]
        else:
            raise ValueError(f"Unknown permutation index.")

    def add_ph_matrices(self, J, R, B=None, Q=None):
        """
        Add port-Hamiltonian (pH) matrices to the Data object.

        This method assigns the provided pH matrices (interconnection J, dissipation R,
        input B, and energy Q) to the internal attributes of the Data object. If a
        matrix is provided as a 2D array, it is broadcasted across all simulations
        by repeating it along a new first dimension. All matrices must be of shape
        (n_sim, n_n * n_dn, n_n * n_dn), except for B which should be of shape
        (n_sim, n_n * n_dn, n_u) when applicable.

        Parameters
        ----------
        J : ndarray
            Interconnection matrix of shape (n_n * n_dn, n_n * n_dn) or
            (n_sim, n_n * n_dn, n_n * n_dn).
        R : ndarray
            Dissipation matrix of shape (n_n * n_dn, n_n * n_dn) or
            (n_sim, n_n * n_dn, n_n * n_dn).
        B : ndarray, optional
            Input matrix of shape (n_n * n_dn, n_u) or
            (n_sim, n_n * n_dn, n_u). Default is None.
        Q : ndarray, optional
            Energy matrix of shape (n_n * n_dn, n_n * n_dn) or
            (n_sim, n_n * n_dn, n_n * n_dn). Default is None.

        Raises
        ------
        AssertionError
            If any provided matrix does not match the expected dimensionality or number of simulations.
        """
        ph_matrices = [J, R, Q, B]
        for i, ph_matrix in enumerate(ph_matrices):
            if ph_matrix is not None and ph_matrix.ndim == 2:
                ph_matrices[i] = np.repeat(
                    ph_matrix[np.newaxis, :, :], self.n_sim, axis=0
                )
            assert ph_matrices[i] is None or (
                ph_matrices[i].shape[0] == self.n_sim
                and ph_matrices[i].shape[1] == self.n_n * self.n_dn
                and ph_matrices[i].ndim == 3
            )
        self.J, self.R, self.Q, self.B = ph_matrices

    def add_system_matrices_from_system_layer(self, system_layer):
        """
        Extract and add port-Hamiltonian system matrices from a given system layer to the Data instance.

        This method retrieves the system matrices (J, R, B, Q, E) from a specified system layer
        using the `get_system_matrices_from_system_layer` method, and integrates them into the
        Data container using `add_ph_matrices`.

        If parameter features (`mu`) are not yet computed but are available as `Mu`, it calls
        `states_to_features()` to generate them.

        Parameters
        ----------
        system_layer : tf.keras.layers.Layer
            The system layer (e.g., DescriptorPHLayer, PHLayer, etc.) from which to extract
            the port-Hamiltonian system matrices.

        Notes
        -----
        - Only the J, R, B, and Q matrices are added to the Data instance. The E matrix is extracted
        but not stored.
        - This method ensures compatibility between learned model parameters and the stored data.
        """
        if self.Mu is not None and self.mu is None:
            self.states_to_features()

        J_ph, R_ph, B_ph, Q_ph, E_ph = self.get_system_matrices_from_system_layer(
            system_layer, mu=self.mu, n_t=self.n_t
        )
        self.add_ph_matrices(J=J_ph, R=R_ph, B=B_ph, Q=Q_ph)

    @staticmethod
    def get_system_matrices_from_system_layer(system_layer, mu, n_t):
        """
        Retrieve system matrices from a given system layer based on its class type.

        This static method extracts the port-Hamiltonian (pH) system matrices from the provided
        `system_layer`, depending on the type of the layer. It supports several types of system layers:
        DescriptorPHQLayer, DescriptorPHLayer, PHQLayer, PHLayer, and LTILayer. Based on the layer,
        it returns the appropriate subset of system matrices (J, R, B, Q, E), with missing matrices
        filled as None if not defined by the layer.

        Parameters
        ----------
        system_layer : tf.keras.layers.Layer
            A system layer from which the pH system matrices will be extracted. Must be an instance of
            DescriptorPHQLayer, DescriptorPHLayer, PHQLayer, PHLayer, or LTILayer.

        mu : tf.Tensor or np.ndarray
            Input parameters to the system layer used for generating the matrices. Typically of shape (n_sim, n_mu).

        n_t : int
            Number of time steps or time points the system layer should be evaluated on.

        Returns
        -------
        tuple
            A tuple containing the system matrices:
            (J, R, B, Q, E), where matrices not available from the given layer are returned as None.
        """
        if isinstance(system_layer, DescriptorPHQLayer):
            J_ph, R_ph, B_ph, Q_ph, E_ph = system_layer.get_system_matrices(mu, n_t=n_t)
        elif isinstance(system_layer, DescriptorPHLayer):
            J_ph, R_ph, B_ph, E_ph = system_layer.get_system_matrices(mu, n_t=n_t)
            Q_ph = None
        elif isinstance(system_layer, PHQLayer):
            J_ph, R_ph, B_ph, Q_ph = system_layer.get_system_matrices(mu, n_t=n_t)
            E_ph = None
        elif isinstance(system_layer, PHLayer) or isinstance(system_layer, LTILayer):
            J_ph, R_ph, B_ph = system_layer.get_system_matrices(mu, n_t=n_t)
            Q_ph = None
            E_ph = None
        else:
            raise NotImplementedError(f"The layer {type(system_layer)} is unknown.")

        return J_ph, R_ph, B_ph, Q_ph, E_ph

    def get_initial_conditions(self):
        """
        Retrieve the initial conditions from the dataset.

        Returns:
        --------
        ndarray
            Initial conditions with shape (n_sim, n_f), where `n_f` is derived from reshaping the data.
        """
        self.x_init = self.X[:, 0, :, :].reshape(self.n_sim, self.n_f)
        return self.x_init

    def train_test_split(self, test_size, seed):
        """
        see Dataset.train_test_split
        """
        raise NotImplementedError(
            "The function train_test_split should only be called from a Dataset instance "
            "and not an instance of the Data class"
        )

    def train_test_split_sim_idx(self, sim_idx_train, sim_idx_test):
        """
        see Dataset.train_test_split_sim_idx
        """
        raise NotImplementedError(
            "The function train_test_split_sim_idx should only be called from a Dataset instance "
            "and not an instance of the Data class"
        )

    def truncate_time(self, trunc_time_ratio):
        """
        Truncates the time values of states for performing time generalization experiments.

        This method shortens the time series of states and associated data based on the given ratio.
        The truncation is applied to time steps, states, time derivatives, inputs, and parameters,
        if they are provided.

        Parameters:
        -----------
            trunc_time_ratio : float
                    Ratio of the time series to retain. A value of 1.0 means no truncation, while a value
                    between 0 and 1 truncates the time series to the specified proportion of the total time steps.

        Returns:
        --------
        None
            The method modifies the internal state of the object to reflect the truncated time series.
        """
        assert trunc_time_ratio >= 0 and trunc_time_ratio <= 1
        self.trunc_time_ratio = trunc_time_ratio
        idx_truncate = int(self.trunc_time_ratio * self.n_t)
        self.t = self.t[:idx_truncate]
        self.X = self.X[:, :idx_truncate]
        self.X_dt = self.X_dt[:, :idx_truncate]
        if self.U is not None:
            self.U = self.U[:, :idx_truncate]
        self.n_t = self.X.shape[1]
        self.states_to_features()

    def cut_time_start_and_end(self, num_cut_idx=5):
        """
        Trim the time series data at the beginning and end to remove potential numerical artifacts.

        This method removes `num_cut_idx` time steps from both the start and end of the dataset.
        This is useful when numerical instabilities or artifacts appear in the time derivatives
        (e.g., `X_dt`) near the boundaries of the simulation time range.

        Parameters
        ----------
        num_cut_idx : int, optional
            Number of time steps to remove from both the beginning and end. Default is 5.

        Notes
        -----
        - It is recommended to apply this method before any scaling or normalization of the data.
        - The internal attributes `t`, `X`, `X_dt`, and optionally `U` are modified in place.
        - The number of time steps `n_t` is updated accordingly.
        - Feature representations are recomputed via `states_to_features()`.
        """
        logging.info(
            f"If the data is cut due to X_dt abnormalities. Use this before scaling the data."
        )
        self.t = self.t[num_cut_idx:-num_cut_idx]
        self.X = self.X[:, num_cut_idx:-num_cut_idx]
        self.X_dt = self.X_dt[:, num_cut_idx:-num_cut_idx]
        if self.U is not None:
            self.U = self.U[:, num_cut_idx:-num_cut_idx]
        self.n_t = self.X.shape[1]
        self.states_to_features()

    def decrease_num_simulations(
        self,
        num_sim: int | None = None,
        seed: int | None = None,
        sim_idx: list[int] | np.ndarray | None = None,
    ):
        """
        Reduce the number of simulations to accelerate training or analysis.

        This method decreases the number of simulations in the dataset by either selecting a specific
        subset (`sim_idx`) or randomly sampling `num_sim` simulations. Useful for debugging, testing,
        or speeding up model training on smaller datasets.

        Parameters
        ----------
        num_sim : int, optional
            Number of simulations to retain. If `sim_idx` is not provided, a random subset of this
            size is chosen. Required if `sim_idx` is not specified.

        seed : int, optional
            Seed for the random number generator to ensure reproducibility when selecting simulations
            randomly. Ignored if `sim_idx` is provided.

        sim_idx : list[int] or np.ndarray, optional
            Explicit list or array of simulation indices to retain. If provided, `num_sim` and `seed`
            are ignored.

        Notes
        -----
        - Modifies the dataset in-place by slicing all internal data attributes (`X`, `X_dt`, `U`, `Mu`,
        `J`, `R`, `Q`, `B`, `Mu_input`) along the simulation axis.
        - Updates `self.n_sim` to reflect the reduced number of simulations.
        - Recomputes features via `self.states_to_features()`.
        - Raises an error if `num_sim` is not provided when `sim_idx` is None.
        """
        if sim_idx is not None:
            if isinstance(sim_idx, np.ndarray):
                assert sim_idx.ndim == 1
                self.n_sim = sim_idx.shape[0]
            elif isinstance(sim_idx, list):
                self.n_sim = len(sim_idx)
            elif isinstance(sim_idx, int):
                self.n_sim = 1
            else:
                raise ValueError(f"Unknown type {type(sim_idx)}of sim_idx.")
        else:
            # default: random
            assert num_sim is not None
            rng = np.random.default_rng(seed=seed)
            sim_idx = rng.choice(self.n_sim, size=num_sim, replace=False)
            self.n_sim = num_sim
        self.X = self.X[sim_idx]
        self.X_dt = self.X_dt[sim_idx]
        self.U = self.U[sim_idx]
        if self.Mu is not None:
            self.Mu = self.Mu[sim_idx]
        if self.J is not None:
            self.J = self.J[sim_idx, :, :]
        if self.R is not None:
            self.R = self.R[sim_idx, :, :]
        if self.Q is not None:
            self.Q = self.Q[sim_idx, :, :]
        if self.B is not None:
            self.B = self.B[sim_idx, :, :]
        if self.Mu_input is not None:
            self.Mu_input = self.Mu_input[sim_idx]
        self.states_to_features()

    def decrease_num_time_steps(self, num_time_steps: int):
        """
        Reduces the number of time steps in the dataset by truncating the time series.

        This method selects a subset of time steps from the original time series to reduce its length.
        It ensures that the number of remaining time steps matches the specified target.

        Parameters:
        -----------
        num_time_steps : int
            The target number of time steps to retain. Must be less than or equal to the current number of time steps.
        """
        self.X, self.X_dt, self.U, self.t, self.n_t = (
            self.decrease_num_time_steps_static(
                num_time_steps=num_time_steps,
                t=self.t,
                X=self.X,
                X_dt=self.X_dt,
                U=self.U,
                n_t=self.n_t,
            )
        )

    @staticmethod
    def decrease_num_time_steps_static(
        num_time_steps: int,
        t: np.ndarray,
        X: np.ndarray | None = None,
        X_dt: np.ndarray | None = None,
        U: np.ndarray | None = None,
        n_t: int | None = None,
    ):
        """
        Reduce the number of time steps in time-series data via 1D interpolation.

        This static method interpolates time-series data to a specified number of time steps,
        which can be useful for downsampling or matching temporal resolution across datasets.

        Parameters
        ----------
        num_time_steps : int
            The desired number of time steps after interpolation.

        t : np.ndarray
            Original time array of shape (n_t,) or (n_t, 1).

        X : np.ndarray, optional
            State array of shape (n_sim, n_t, n_n, n_dn). Interpolated along the time axis if provided.

        X_dt : np.ndarray, optional
            Time derivative of the state array, same shape as X. Interpolated if provided.

        U : np.ndarray, optional
            Input array of shape (n_sim, n_t, n_u). Interpolated if provided.

        n_t : int, optional
            Original number of time steps. If not provided, inferred from X.

        Returns
        -------
        tuple
            A tuple (X, X_dt, U, t, n_t) containing the interpolated arrays and new time information.

        Notes
        -----
        - All interpolation is linear along the time axis.
        - If `X`, `X_dt`, or `U` is not provided, it will be returned as `None`.
        """
        if n_t is None:
            n_t = X.shape[1]
        assert num_time_steps <= n_t  # interpolate
        # create interpolation functions and interpolate data at new time steps
        t_new = np.linspace(t[0], t[-1], num_time_steps).ravel()
        if X is not None:
            f_interp_X = interp1d(t.ravel(), X, axis=1)
            X = f_interp_X(t_new)
        if X_dt is not None:
            f_interp_X_dt = interp1d(t.ravel(), X_dt, axis=1)
            X_dt = f_interp_X_dt(t_new)
        if U is not None:
            f_interp_U = interp1d(t.ravel(), U, axis=1)
            U = f_interp_U(t_new)
        t = t_new[:, np.newaxis]
        n_t = t_new.shape[0]

        return X, X_dt, U, t, n_t

    @staticmethod
    def save_data(data_path, t, X, U, Mu=None, Mu_input: np.ndarray = None):
        """
        Saves time steps, state data, and inputs to a compressed .npz file.

        This method saves the provided time steps, state data, inputs, and optional parameters to a `.npz` file
        for efficient storage and retrieval.

        Parameters:
        -----------
        data_path : str
            The path to the `.npz` file where the data will be saved.
        t : ndarray
            Array of time steps with shape (n_t,).
        X : ndarray
            Array of system states at all time steps with shape (n_sim, n_t, n_n, n_dn).
        U : ndarray
            Array of system inputs with shape (n_sim, n_t, n_u).
        Mu : ndarray, optional
            Array of parameters with shape (n_sim, n_mu). Default is None. If provided, it will be saved along with the other data.
        Mu_input : ndarray, optional
            Array of parameters that were used to define the input U. Default is None. If provided, it
            will be saved along with the other data.

        Returns:
        --------
        None
            This method does not return any value. It saves the data to the specified file path.
        """
        if Mu is not None:
            if Mu_input is not None:
                np.savez_compressed(data_path, t=t, X=X, U=U, Mu=Mu, Mu_input=Mu_input)
            else:
                np.savez_compressed(data_path, t=t, X=X, U=U, Mu=Mu)
        else:
            if Mu_input is not None:
                np.savez_compressed(data_path, t=t, X=X, U=U, Mu_input=Mu_input)
            else:
                np.savez_compressed(data_path, t=t, X=X, U=U)

    @classmethod
    def from_data(cls, data_path, **kwargs):
        """
        Loads a dataset from a .npz file and creates an instance of the class.

        This class method reads time steps, state data, and inputs from a `.npz` file and initializes
        an instance of the class using the loaded data. The file should contain arrays for time steps (`t`),
        states (`X`), and inputs (`U`), and optionally other parameters.

        Parameters:
        -----------
        data_path : str
            Path to the `.npz` file containing the dataset.
        **kwargs : keyword arguments
            Additional parameters to pass to the class constructor.

        Returns:
        --------
        instance of cls
            An instance of the class initialized with the data from the `.npz` file and any additional parameters provided.

        Notes:
        ------
        The `.npz` file should contain the following arrays:
        - 't' : ndarray with shape (n_t,)
        - 'X' : ndarray with shape (n_sim, n_t, n_n, n_dn)
        - 'U' : ndarray with shape (n_sim, n_t, n_u)
        """
        # search for file in data_path directory if no file name is given
        data_dict = cls.read_data_from_npz(data_path)

        return cls(**data_dict, **kwargs)

    @staticmethod
    def read_data_from_npz(data_path, num_time_steps: int | None = None):
        """
        Reads data from a .npz file and returns it as a dictionary.

        Parameters:
        -----------
        data_path : str
            Path to the .npz file or the directory containing the .npz file. If a directory
            is provided, the method searches for the first .npz file in the directory.

        num_time_steps : int, optional
            The number of time steps to retain. If provided, the data will be truncated to this number
            of time steps. Default is `None`.

        Returns:
        --------
        dict
            Dictionary containing the following keys:
            - 't': ndarray, array of time steps.
            - 'X': ndarray, states array.
            - 'X_dt': ndarray, time derivatives of the states (optional, may be None).
            - 'U': ndarray, input array (optional, may be None).
            - 'Mu': ndarray, parameters array (optional, may be None).
            - 'J': ndarray, pH interconnection matrix (optional, may be None).
            - 'R': ndarray, pH dissipation matrix (optional, may be None).
            - 'Q': ndarray, pH energy matrix (optional, may be None).
            - 'B': ndarray, pH input matrix (optional, may be None).
        """

        # search for file in data_path directory if no file name is given
        if os.path.isdir(data_path):
            npz_file_found = False
            for file in os.listdir(data_path):
                if file.endswith(".npz"):
                    file_name = file
                    logging.info(
                        f"Found file {file_name} in {data_path} which will be used subsequently. Use direct path to file if you want to specify a different data file."
                    )
                    data_path = os.path.join(data_path, file_name)
                    npz_file_found = True
                    break  # choose first .npz file that occurs
            if not npz_file_found:
                raise ValueError(f"No .npz file found in {data_path}.")

        # load data
        logging.info(f"Loading data from cache: {data_path}")
        data = np.load(data_path, allow_pickle=True)
        X = data["X"]
        t = data["t"]
        U, X_dt, Mu, J, R, Q, B, Mu_input = [None] * 8
        if "X_dt" in data.keys():
            X_dt = data["X_dt"] if np.any(data["X_dt"]) else None
        if "U" in data.keys():
            U = data["U"] if np.any(data["U"]) else None
        if "Mu" in data.keys():
            Mu = data["Mu"] if np.any(data["Mu"]) else None
        # matrices in format (r,r,n_sim)
        if "J" in data.keys():
            J = data["J"]
        if "R" in data.keys():
            R = data["R"]
        if "Q" in data.keys():
            Q = data["Q"]
        if "B" in data.keys():
            # B matrix in  format (r,n_u,n_sim)
            B = data["B"]
        if "Mu_input" in data.keys():
            Mu_input = data["Mu_input"]

        if num_time_steps is not None:
            X, X_dt, U, t, _ = Data.decrease_num_time_steps_static(
                num_time_steps=num_time_steps, t=t, X=X, X_dt=X_dt, U=U, n_t=X.shape[1]
            )

        data_dict = {
            "t": t,
            "X": X,
            "X_dt": X_dt,
            "U": U,
            "Mu": Mu,
            "J": J,
            "R": R,
            "Q": Q,
            "B": B,
            "Mu_input": Mu_input,
        }

        return data_dict

    def states_to_features(self):
        """
        Transforms the state array into a feature array for identification purposes.

        This method reshapes the state array `X` and its time derivatives `X_dt` into feature arrays
        that are required for system identification. It also reshapes the input array `U` and parameter
        array `Mu` if they are provided.

        The transformation results in:
        - `x`: a feature array of shape (n_s, n_f), where:
            - `n_s` is the number of samples (`n_sim * n_t`)
            - `n_f` is the number of features (`n_n * n_dn`)
        - `u`: reshaped input array of shape (n_s, n_u) if `U` is provided
        - `mu`: reshaped parameters array of shape (n_s, n_mu) if `Mu` is provided

        Returns:
        --------
        None
            The method updates the internal state of the object with the transformed feature arrays.
        """
        # training data
        self.n_s = self.n_sim * self.n_t
        self.n_f = self.n_n * self.n_dn
        self.x, self.dx_dt = reshape_states_to_features(self.X, self.X_dt)
        assert self.x.shape == (self.n_s, self.n_f)
        if self.U is not None:
            self.u = reshape_inputs_to_features(self.U)

        # parameters
        if self.Mu is not None:
            self.mu = np.repeat(self.Mu[:, np.newaxis], self.n_t, axis=1)
            self.mu = self.mu.reshape(self.n_sim * self.n_t, self.n_mu)

    def features_to_states(self):
        """
        Transforms the feature array back into the state array required for validation.

        This method reshapes the feature arrays `x` and `dx_dt` back into the original state arrays `X`
        and `X_dt`. The transformation restores the dimensions to match the number of simulations,
        time steps, nodes, and degrees of freedom per node.

        Parameters:
        -----------
        None
            The method uses internal attributes `x`, `dx_dt`, `n_sim`, `n_t`, `n_n`, and `n_dn` to perform
            the transformation.

        Returns:
        --------
        None
            The method updates the internal state of the object with the reshaped state arrays `X`
            and `X_dt`.
        """
        self.X = reshape_features_to_states(
            self.x, self.n_sim, self.n_t, n_n=self.n_n, n_dn=self.n_dn
        )
        self.X_dt = reshape_features_to_states(
            self.dx_dt, self.n_sim, self.n_t, n_n=self.n_n, n_dn=self.n_dn
        )
        assert self.X.shape == (self.n_sim, self.n_t, self.n_n, self.n_dn)

    def split_state_into_domains(self, domain_split_vals=None):
        """
        Splits the state array into different domains based on the specified dimensions.

        This method divides the state array `X` into multiple domains according to the given
        `domain_split_vals`. Each domain corresponds to a subset of degrees of freedom (dofs), and
        the sum of these values must equal the total number of dofs per node (`self.n_dn`).

        Parameters:
        -----------
        domain_split_vals : list of int, optional
            List specifying the number of dofs in each domain. For example, `[1, 2, 2]` indicates three
            domains with 1, 2, and 2 dofs, respectively. The sum of these values must equal `self.n_dn`.
            If `None`, the entire state is considered as a single domain.

        Returns:
        --------
        list of ndarray
            A list where each element is a state array corresponding to a specific domain. The length
            of the list equals the number of domains specified by `domain_split_vals`.
        """
        if domain_split_vals is None:
            domain_split_vals = [self.n_dn]

        assert sum(domain_split_vals) == self.n_dn

        X_dom_list = []
        n_dom = 0
        for i_dom in range(len(domain_split_vals)):
            X_dom_list.append(self.X[:, :, :, n_dom : n_dom + domain_split_vals[i_dom]])
            n_dom += domain_split_vals[i_dom]
        return X_dom_list

    def split_input_into_domains(self, input_domain_split_vals=None):
        """
        TODO: write header
        """
        if input_domain_split_vals is None:
            input_domain_split_vals = [self.n_u]

        assert sum(input_domain_split_vals) == self.n_u

        U_dom_list = []
        n_dom = 0
        for i_dom in range(len(input_domain_split_vals)):
            U_dom_list.append(
                self.U[..., n_dom : n_dom + input_domain_split_vals[i_dom]]
            )
            n_dom += input_domain_split_vals[i_dom]
        return U_dom_list

    def calculate_eigenvalues(
        self, result_dir: str, save_to_csv: bool = False, save_name: str = "eigenvalues"
    ):
        """
        Calculate the eigenvalues of the system matrix A and optionally save them to a CSV file.

        This method computes the eigenvalues of the matrix A, where A is defined as (J - R) or (J - R) @ Q,
        depending on whether the energy matrix Q is provided. The maximum eigenvalues for each scenario are
        printed to the console, and optionally, both the real and imaginary parts of the eigenvalues are saved
        to a CSV file for further analysis.

        Parameters:
        -----------
        result_dir : str
            The directory where the eigenvalues will be saved if `save_to_csv` is `True`.

        save_to_csv : bool, optional
            If `True`, the eigenvalues will be saved to a CSV file. Default is `False`.

        save_name : str, optional
            The name of the CSV file to which the eigenvalues will be saved (if `save_to_csv` is `True`).
            Default is "eigenvalues".

        Returns:
        --------
        eig_vals : ndarray
            A 2D array of eigenvalues, with shape (n_sim, n_eigenvalues), where `n_sim` is the number of simulations
            and `n_eigenvalues` is the number of eigenvalues per simulation.

        Notes:
        ------
        - If the energy matrix Q is not provided, the matrix A is calculated as (J - R).
        - If Q is provided, the matrix A is calculated as (J - R) @ Q.
        - The maximum eigenvalue for each simulation is printed to the console.
        - The eigenvalues are saved in a CSV file with the real and imaginary parts as separate columns if `save_to_csv` is `True`.

        Raises:
        -------
        ValueError
            If the system matrices J or R are not defined.
        """
        if self.Q is None:
            A = self.J - self.R
        else:
            A = (self.J - self.R) @ self.Q
        eig_vals = np.array([np.linalg.eig(A_).eigenvalues for A_ in A])
        max_eigs = np.array(np.real(eig_vals).max(axis=1))

        print(f"The maximum eigenvalues of A for all scenarios are:\n {max_eigs}")
        # max_eigs_pred = np.array(
        #     [np.real(np.linalg.eig(A_).eigenvalues).max() for A_ in A_pred]
        # )

        # # plot eigenvalues on imaginary axis
        # eigs_pred = np.array([np.linalg.eig(A_).eigenvalues for A_ in A_pred])

        # save real and imaginary parts of eigenvalues to csv
        if save_to_csv:
            header = "".join(
                [f"sim{i}_eigs_real," for i in range(eig_vals.shape[0])]
            ) + "".join([f"sim{i}_eigs_imag," for i in range(eig_vals.shape[0])])
            np.savetxt(
                os.path.join(result_dir, f"{save_name}.csv"),
                np.concatenate([eig_vals.real, eig_vals.imag], axis=0).T,
                delimiter=",",
                header=header,
                comments="",
            )
        return eig_vals

    def calculate_errors(self, ph_identified_data_instance, domain_split_vals=None):
        """
        Calculate and store RMS and latent errors between true and predicted states.

        This method computes the Root Mean Square (RMS) errors for the states and latent variables between
        the true and predicted values. It calculates the RMS errors for each domain, computes the mean RMS
        error across all simulations, and, if available, computes the latent variable errors.

        Parameters:
        -----------
        ph_identified_data_instance : Data
            An instance of the Data class containing the predicted state variables and latent variables
            (if applicable). This instance is used to compute errors against the true states stored in
            the current instance.

        domain_split_vals : list of int, optional
            List specifying the number of degrees of freedom (DOFs) for each domain. If provided, it splits
            the states into domains for more granular error analysis.

        Returns:
        --------
        None
            The method updates internal attributes to store the computed state errors and latent errors
            (if latent variables are present).

        Notes:
        ------
        - `self.state_error_list` contains the RMS errors for each domain.
        - `self.state_error_mean` is the mean RMS error averaged over all simulations.
        - `self.latent_error` and `self.latent_error_mean` are computed only if the `ph_identified_data_instance`
        contains latent variables (`Z`).
        """
        # RMS
        self.state_error_list = self.calculate_state_errors(
            ph_identified_data_instance, domain_split_vals=domain_split_vals
        )
        # mean over simulations
        self.state_error_mean = np.array(self.state_error_list).mean()
        if hasattr(ph_identified_data_instance, "Z"):
            self.latent_error = self.calculate_latent_errors(
                ph_identified_data_instance
            )
            self.latent_error_mean = self.latent_error.mean()
        else:
            self.latent_error = None
            self.latent_error_mean = None

    def calculate_latent_errors(self, ph_identified_data_instance):
        """
        Compute RMS errors for latent variables between true and predicted values.

        This method calculates the Root Mean Square (RMS) error between the true latent variables (`Z`)
        and the predicted latent variables (`Z_ph`) from an identified data instance.

        Parameters:
        -----------
        ph_identified_data_instance : instance of Data
            An instance of the Data class containing the predicted latent variables (`Z_ph`) and
            true latent variables (`Z`) used for comparison.

        Returns:
        --------
        ndarray
            Array of RMS errors for the latent variables, with shape consistent with the latent variable dimensions.
        """
        Z = np.expand_dims(ph_identified_data_instance.Z, axis=3)
        Z_ph = np.expand_dims(ph_identified_data_instance.Z_ph, axis=3)
        latent_error = self.calculate_rms_error(Z, Z_ph)
        return latent_error

    def calculate_state_errors(
        self, ph_identified_data_instance, domain_split_vals=None
    ):
        """
        Calculate the normalized RMS error and the relative error between true states and predicted states.

        This method computes errors by comparing the true state values (`self.X`) with the predicted
        state values from an identified data instance (`ph_identified_data_instance`). Errors are
        calculated for each domain if specified.

        Parameters:
        -----------
        ph_identified_data_instance : instance of Data
            An instance of the Data class containing the predicted states used for comparison.

        domain_split_vals : list of int, optional
            List specifying the number of degrees of freedom in each domain. This is used to split the
            state arrays into different domains for error calculation. The sum of these values must
            match the total number of dofs per node (`self.n_dn`). If `None`, the entire state is
            considered as a single domain.

        Returns:
        --------
        list of ndarray
            A list of normalized RMS errors for each domain. Each element of the list represents the
            RMS error for a specific domain, calculated as the difference between the true and predicted
            states, normalized by the total number of samples and features.
        """
        X_dom_list = self.split_state_into_domains(domain_split_vals=domain_split_vals)
        X_dom_list_id = ph_identified_data_instance.split_state_into_domains(
            domain_split_vals=domain_split_vals
        )

        # loop over domains
        state_error_list = []
        for X, X_id in zip(X_dom_list, X_dom_list_id):
            norm_rms_error = self.calculate_rms_error(X, X_id)
            state_error_list.append(norm_rms_error)
        return state_error_list

    @staticmethod
    def calculate_rms_error(X_or_Z, X_or_Z_id):
        """
        Calculates the root mean square (RMS) error between the given dataset and its identified counterpart.

        Parameters:
        -----------
        X_or_Z : ndarray
            Array of states (X) or latent states (Z) with shape (n_sim, n_t, n_n, n_dn) or (n_sim, n_t, r) respectively.
        X_or_Z_id : ndarray
            Array of identified states (X_id) or identified latent states (Z_id) with the same shape as X_or_Z.

        Returns:
        --------
        norm_rms_error : ndarray
            Array of normalized RMS errors with shape (n_sim, n_t).
        """
        assert X_or_Z.shape == X_or_Z_id.shape
        assert X_or_Z.ndim == 4
        # norm over node dofs (space directions)
        X_error = np.linalg.norm(X_or_Z - X_or_Z_id, axis=3)
        X_norm = np.linalg.norm(X_or_Z, axis=3)
        # calculate norm over all nodes and relative error over the time
        norm_rms_error = (
            np.linalg.norm(X_error, axis=2)
            / np.linalg.norm(X_norm, axis=2).mean(axis=1)[:, np.newaxis]
        )
        return norm_rms_error

        # for j in range(6):
        #     plt.figure()
        #     for i in range(6):
        #         plt.plot(X_error[j, :,i], "gray")
        #     X_error_mean = np.linalg.norm(X_error, axis=2)
        #     plt.plot(X_error_mean[j, :])
        #     plt.show()

    def remove_mu(self):
        """
        Removes the parameters `mu` and `Mu` from the data.

        This method resets the `mu` and `Mu` attributes to `None` and sets `n_mu` to 0, effectively
        removing the parameters from the data. This can be useful when you no longer need the parameters
        for further computations or if you want to prepare the data for a different set of parameters.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Notes:
        ------
        - After calling this method, the `mu`, `Mu`, and `n_mu` attributes will be reset, and any future computations
        depending on these parameters will need to handle the absence of `mu` and `Mu`.
        """
        self.mu = None
        self.Mu = None
        self.n_mu = 0

    def scale_all(
        self,
        scaling_values=None,
        domain_split_vals=None,
        u_train_bounds=None,
        u_desired_bounds=[-1, 1],
        mu_train_bounds=None,
        mu_desired_bounds=[-1, 1],
    ):
        """
        Scales the states (X), inputs (U), and parameters (Mu) of the system.

        This method provides a unified way to scale the state array (`X`), input array (`U`), and parameter array (`Mu`)
        based on the provided scaling values and bounds. It first scales the state data using the `scale_X` method and
        then scales the inputs and parameters (if available) using their respective scaling methods.

        Parameters:
        -----------
        scaling_values : list of float, optional
            Scalar values used to scale each domain in the state array, as described in `scale_X`. If `None`, the scaling
            is performed using the maximum value of each domain.

        domain_split_vals : list of int, optional
            List of integers specifying the degrees of freedom (DOFs) for each domain in the state array, as described in
            `scale_X`. If `None`, the state array is treated as a single domain.

        u_train_bounds : list of float, optional
            Training bounds for the input array (`U`). If `None`, no scaling is applied to `U`.

        u_desired_bounds : list of float, optional
            Desired bounds for the input array (`U`). Default is [-1, 1].

        mu_train_bounds : list of float, optional
            Training bounds for the parameter array (`Mu`). If `None`, no scaling is applied to `Mu`.

        mu_desired_bounds : list of float, optional
            Desired bounds for the parameter array (`Mu`). Default is [-1, 1].

        Returns:
        --------
        None
            The method updates the internal state to reflect the scaled data, including the state (`X`), inputs (`U`),
            and parameters (`Mu`). It also sets `self.is_scaled` to True.

        Notes:
        ------
        - This method scales all three components (states, inputs, and parameters) in a single call.
        - If `scaling_values` and `domain_split_vals` are provided, the scaling is done according to the specified values.
        - The method assumes that the scaling operations for each component are handled by the respective methods (`scale_X`,
        `scale_U`, and `scale_Mu`).
        """
        self.scale_X(scaling_values=scaling_values, domain_split_vals=domain_split_vals)
        if self.U is not None:
            self.scale_U(u_train_bounds=u_train_bounds, desired_bounds=u_desired_bounds)
        if self.Mu is not None:
            self.scale_Mu(
                mu_train_bounds=mu_train_bounds, desired_bounds=mu_desired_bounds
            )

    def scale_X(self, scaling_values=None, domain_split_vals=None):
        """
        Scale the state array based on specified scaling values.

        This method scales the state array `X` with shape (n_sim, n_t, n_n, n_dn) and afterwards the feature array `x`
        with shape (n_t * n_s, n) using provided scaling values. It can handle multiple domains if specified
        by `domain_split_vals`. If scaling values are not provided, it defaults to scaling by the maximum value
        in each domain.

        Parameters:
        -----------
        scaling_values : list of float, optional
            Scalar values used to scale each domain, as defined by `domain_split_vals`. If `None`, scaling
            is performed by the maximum value of each domain.

        domain_split_vals : list of int, optional
            List of integers specifying the number of degrees of freedom (DOFs) for each domain. The sum of
            these values must equal `n_dn`. If `None`, the data is treated as a single domain.

        Returns:
        --------
        None
            The method updates the internal state to reflect the scaled data and sets `self.is_scaled` to True.

        Notes:
        ------
        - The method performs scaling for both the state data (`X`) and its time derivatives (`X_dt`).
        - If `scaling_values` are not provided, the maximum value for each domain is used for scaling.
        - The method assumes that the domains defined by `domain_split_vals` add up to the total number of DOFs (`n_dn`).
        - After scaling, the feature representation of states is updated using `self.states_to_features()`.
        """
        if domain_split_vals is None:
            # use just one domain as default
            domain_split_vals = [self.n_dn]

        assert sum(domain_split_vals) == self.n_dn

        if scaling_values is None:
            logging.warning(
                "No scaling values are given. Scaling with maximum value. Only define the scale values for the training data set. "
                "Don't call this function without scaling_values for test data."
            )
            # split data into domains
            X_dom_list = self.split_state_into_domains(
                domain_split_vals=domain_split_vals
            )
            current_idx = 0
            scaling_values_list = []
            for i, X in enumerate(X_dom_list):
                scaling_values_list.append(np.max(np.abs(X)))
        else:
            scaling_values_list = scaling_values

        # reformulate scaling values into array format
        if len(scaling_values_list) == len(domain_split_vals):
            # repeat scaling values for all domain entries
            current_idx = 0
            scaling_values = np.zeros((self.n_dn, 1))
            for i in range(len(domain_split_vals)):
                scaling_values[current_idx : current_idx + domain_split_vals[i], :] = (
                    scaling_values_list[i] * np.ones((domain_split_vals[i], 1))
                )
                current_idx += domain_split_vals[i]
        elif len(scaling_values_list) == self.n_dn:
            # transform to numpy array
            scaling_values = np.array(scaling_values)
        else:
            raise ValueError(f"Scaling values are not given in the right format.")

        self.scaling_values = scaling_values
        self.is_scaled = True

        # scale states
        for i_n_dn in range(self.n_dn):
            self.X[:, :, :, i_n_dn] = (
                self.X[:, :, :, i_n_dn] / self.scaling_values[i_n_dn]
            )
            self.X_dt[:, :, :, i_n_dn] = (
                self.X_dt[:, :, :, i_n_dn] / self.scaling_values[i_n_dn]
            )

        # transform also state values if they have been calculated already
        self.states_to_features()

    def rescale_X(self):
        """
        Undo the scaling of states and time derivatives.

        This method reverses the scaling applied to the state array `X` with shape (n_sim, n_t, n_n, n_dn)
        and the time derivatives `X_dt`. It must be called after `scale_X` has been performed to restore the
        original data values. The method assumes that scaling has been previously applied and will raise an
        error if no scaling has been performed.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            The method updates the internal state to reflect the rescaled data and sets `self.is_scaled` to False.
        """
        if self.is_scaled is False:
            raise ValueError("Data has not been scaled yet")

        self.is_scaled = False

        # state data
        for i_n_dn in range(self.n_dn):
            self.X[:, :, :, i_n_dn] = (
                self.X[:, :, :, i_n_dn] * self.scaling_values[i_n_dn]
            )
            self.X_dt[:, :, :, i_n_dn] = (
                self.X_dt[:, :, :, i_n_dn] * self.scaling_values[i_n_dn]
            )

        # transform also state values if they have been calculated already
        self.states_to_features()

    def scale_U(
        self,
        u_train_bounds=None,
        desired_bounds=[-1, 1],
    ):
        """
        Scale input values to a specified range.

        This method scales the input values `U` to the range defined by `desired_bounds`, which is typically
        [-1, 1]. Each dimension of the input is scaled individually. The scaling factors are either provided
        through `u_train_bounds` or computed based on the provided data.

        Parameters:
        -----------
        u_train_bounds : np.ndarray, optional
            Scaling factors obtained from the training parameter dataset. Expected shape is (2, n_u), where
            n_u is the number of input dimensions. If provided, these factors are used for scaling the input values.

        desired_bounds : list or scalar, optional
            Desired range for the scaled inputs. Can be:
            - A list of two values (e.g., [-1, 1]) to specify the lower and upper bounds.
            - A scalar to scale all input values by this value.
            - The string "max" to scale all input values by the maximum value observed.

        Returns:
        --------
        None
            The method updates the internal state by scaling the input values `U`, setting `self.u_train_bounds`,
            and updating `self.desired_bounds_u`. The scaled inputs are reshaped into feature format.
        """
        if self.u_train_bounds is not None:
            logging.warning(
                "Inputs u have already been scaled. The scaled u values are scaled again."
            )
        self.U, self.u_train_bounds, self.desired_bounds_u = self.scale_quantity(
            self.U, u_train_bounds, desired_bounds
        )
        self.u = reshape_inputs_to_features(self.U)

        return self.U, self.u_train_bounds

    def scale_U_domain_wise(
        self,
        input_domain_split_vals: list[int] = None,
        input_scaling_values: list[float] = None,
    ):
        """
        Scales the input data `U` domain-wise using either provided or computed scaling values.

        This method divides the input dimension into specified domains and applies domain-wise scaling.
        If no scaling values are provided, it automatically computes the scaling values as the maximum absolute
        value in each domain.

        Parameters:
        -----------
        input_domain_split_vals : list[int], optional
            A list of integers indicating how the input dimensions (n_u) are split across different domains.
            The sum of this list must equal `self.n_u`. If None, all inputs are treated as a single domain.

        input_scaling_values : list[float], optional
            A list of scaling values corresponding to each domain or each input dimension. If not provided,
            the function will compute the scaling values from the current data (intended for training only).

        Returns:
        --------
        tuple:
            - self.U (np.ndarray): The scaled input tensor.
            - self.input_scaling_values (np.ndarray): The array of scaling values used.
        """
        if input_domain_split_vals is None:
            # use just one domain as default
            input_domain_split_vals = [self.n_u]
        assert sum(input_domain_split_vals) == self.n_u

        if input_scaling_values is None:
            logging.warning(
                "No scaling values are given. Scaling with maximum value. Only define the scale values for the training data set. "
                "Don't call this function without scaling_values for test data."
            )
            # split data into domains
            U_dom_list = self.split_input_into_domains(
                input_domain_split_vals=input_domain_split_vals
            )
            current_idx = 0
            scaling_values_list = []
            for i, U in enumerate(U_dom_list):
                scaling_values_list.append(np.max(np.abs(U)))
        else:
            scaling_values_list = input_scaling_values

            # reformulate scaling values into array format
        if len(scaling_values_list) == len(input_domain_split_vals):
            # repeat scaling values for all domain entries
            current_idx = 0
            input_scaling_values = np.zeros((self.n_u, 1))
            for i in range(len(input_domain_split_vals)):
                input_scaling_values[
                    current_idx : current_idx + input_domain_split_vals[i], :
                ] = scaling_values_list[i] * np.ones((input_domain_split_vals[i], 1))
                current_idx += input_domain_split_vals[i]
        elif len(scaling_values_list) == self.n_u:
            # transform to numpy array
            input_scaling_values = np.array(input_scaling_values)
        else:
            raise ValueError(f"Scaling values are not given in the right format.")

        self.input_scaling_values = input_scaling_values
        self.input_is_scaled = True

        # scale states
        for i_n_u in range(self.n_u):
            self.U[..., i_n_u] = self.U[..., i_n_u] / self.input_scaling_values[i_n_u]

        self.u = reshape_inputs_to_features(self.U)

        return self.U, self.input_scaling_values

    def scale_Mu(self, mu_train_bounds=None, desired_bounds=[-1, 1]):
        """
        Scale parameter values to the specified range and return the scaled values along with scaling factors.

        This method scales the parameter values in `Mu` to fit within the `desired_bounds` range. Each parameter
        dimension is scaled individually. Scaling can be based on maximum values, a scalar, or provided training
        bounds.

        Parameters:
        -----------
        desired_bounds : list, scalar, or "max", optional
            Desired range for the scaled values. It can be:
            - A list of two values (e.g., [-1, 1]) specifying the lower and upper bounds.
            - A scalar, where all values are scaled by this single value.
            - The string "max", where scaling is based on the maximum value in `Mu`.

        mu_train_bounds : np.ndarray, optional
            Scaling factors obtained from the training parameter dataset. Expected shape is (2, n_mu), where
            n_mu is the number of parameter dimensions in `Mu`. If not provided, scaling factors are computed from
            the data.

        Returns:
        --------
        scaled_Mu : np.ndarray
            The scaled values of `Mu` with the same shape as the input.

        mu_train_bounds : np.ndarray
            The computed or provided scaling factors used for scaling.
        """
        if self.mu_train_bounds is not None:
            logging.warning(
                "Parameters mu have already been scaled. The scaled mu values are scaled again."
            )

        self.Mu, self.mu_train_bounds, self.desired_bounds_mu = self.scale_quantity(
            self.Mu, mu_train_bounds, desired_bounds
        )

        return self.Mu, self.mu_train_bounds

    def scale_quantity(self, Quantity, train_bounds=None, desired_bounds=[-1, 1]):
        """
        Scale a given quantity to a specified range and return the scaled values along with scaling factors.

        This method scales the input `Quantity` values to fit within the `desired_bounds` range. Each dimension
        of the input is scaled individually. The scaling can be based on maximum values, a scalar, or training
        bounds if provided.

        Parameters:
        -----------
        Quantity : np.ndarray
            The data to be scaled. It can be of shape (n_sim, n_quantity) for time-independent data or
            (n_sim, n_t, n_quantity) for time-dependent data, where:
            - n_sim: number of simulations
            - n_t: number of time steps
            - n_quantity: number of quantities to be scaled

        train_bounds : np.ndarray, optional
            Scaling factors obtained from the training parameter dataset. Expected shape is (2, n_q), where
            n_q is the number of dimensions in `Quantity`. If not provided, scaling factors are computed from
            the data.

        desired_bounds : list, scalar, or "max", optional
            Desired range for the scaled values. It can be:
            - A list of two values (e.g., [-1, 1]) specifying the lower and upper bounds.
            - A scalar, where all values are scaled by this single value.
            - The string "max", where scaling is based on the maximum value in `Quantity`.

        Returns:
        --------
        scaled_Quantity : np.ndarray
            The scaled values of `Quantity` with the same shape as the input.

        train_bounds : np.ndarray
            The computed or provided scaling factors used for scaling.

        desired_bounds : list
            The bounds used for scaling, as provided or computed.
        """
        # in case of (n_sim x n_quantity) (time-independent) we take the minima and maxima over axis=0
        if Quantity.ndim == 2:
            axes = (0, 1)
        # in case of (n_sim x n_t x n_quantity) (time-dependent) we take the minima and maxima over axis=(0, 1)
        elif Quantity.ndim == 3:
            axes = ((0, 1), 2)

        # ensure Quantity is of dtype float64 - leads to error if scaled to e.g. [0,1] because int does not allow for 0.x values
        Quantity = Quantity.astype("float64")

        n_q = Quantity.shape[-1]
        if desired_bounds == "max":
            # scale with maximum value
            u_max = np.max(Quantity)
            desired_bounds = u_max

        if np.isscalar(desired_bounds):
            # scale with value defined in desired_bounds
            Quantity = Quantity / desired_bounds
            train_bounds = None
        else:
            # scale to lower and upper bound given by desired_bounds
            if train_bounds is None:
                # define training bounds from training data
                min = np.min(Quantity, axis=axes[0])[:, np.newaxis].T
                max = np.max(Quantity, axis=axes[0])[:, np.newaxis].T
                train_bounds = np.concatenate((min, max), axis=0)
            else:
                train_bounds = train_bounds

            for i_q in range(n_q):
                if abs(train_bounds[1, i_q] - train_bounds[0, i_q]) < 1e-12:
                    if self.n_sim == 1:
                        logging.warning(
                            f"Just one parameter scenario is given. Scale parameter values over axis=2"
                        )
                        # get min and max over axis=1
                        min = np.min(Quantity, axis=axes[1])[:, np.newaxis].T
                        max = np.max(Quantity, axis=axes[1])[:, np.newaxis].T
                        train_bounds = np.concatenate((min, max), axis=0)

                        Quantity[0] = (
                            (desired_bounds[1] - desired_bounds[0])
                            / (train_bounds[1, 0] - train_bounds[0, 0])
                        ) * (Quantity[0] - train_bounds[0, 0]) + desired_bounds[0]
                        break
                    else:
                        logging.warning(
                            f"min and max are too close to each other. Quantity {i_q} is not scaled"
                        )
                        pass
                else:
                    Quantity[..., i_q] = (
                        (desired_bounds[1] - desired_bounds[0])
                        / (train_bounds[1, i_q] - train_bounds[0, i_q])
                    ) * (Quantity[..., i_q] - train_bounds[0, i_q]) + desired_bounds[0]

        # update feature format
        return Quantity, train_bounds, desired_bounds

    def filter_data(self, window=10, order=3, interp_equidis_t=False):
        """
        Apply Savitzky-Golay filtering to the state data to smooth it and compute the time derivatives.

        This method filters the state data `X` and its derivative `X_dt` using the Savitzky-Golay filter to reduce noise and smooth the
        data. If `interp_equidis_t` is set to True, the method will first interpolate the data to equally spaced
        time points before applying the filter.

        Parameters:
        -----------
        window : int, optional
            The length of the filter window (i.e., the number of points used to calculate the smoothing).
            It must be an odd integer. Default is 10.

        order : int, optional
            The order of the polynomial used to fit the samples. It must be less than the window length.
            Default is 3.

        interp_equidis_t : bool, optional
            If True, interpolate the data to equally spaced time points before filtering.
            Default is False.
        """

        # interpolate at equally spaced time points
        if interp_equidis_t:
            logging.info(f"interpolating state at equidistant time points.")
            f_interp = interp1d(self.t.ravel(), self.X, axis=1)
            self.t = np.linspace(self.t[0], self.t[-1], self.n_t)
            self.X = f_interp(self.t.ravel())

        delta_t = self.t[1] - self.t[0]
        self.X = savgol_filter(
            self.X,
            window_length=window,
            polyorder=order,
            deriv=0,
            axis=1,
            delta=delta_t,
        )
        self.X_dt = savgol_filter(
            self.X,
            window_length=window,
            polyorder=order,
            deriv=1,
            axis=1,
            delta=delta_t,
        )

    def save_state_traj_as_csv(
        self, path, dof=0, second_oder=False, filename="state_trajectories"
    ):
        """
        Save the state trajectories and their time derivatives as CSV files.

        This method saves the state trajectories and, if applicable, their time derivatives
        to CSV files. The method can handle both first-order and second-order systems,
        with the option to exclude redundant derivative data for second-order systems.

        Parameters:
        -----------
        path : str
            Directory path where the CSV files will be saved.

        dof : int, optional
            Degree of freedom (dof) index to save. Default is 0.

        second_order : bool, optional
            If True, the system is treated as second-order, and additional processing is applied
            to handle state derivatives accordingly. Default is False.

        filename : str, optional
            Base name for the output CSV files. Default is "state_trajectories".
        """
        # for second order systems the first entries of X_dt match the last of X and are not needed
        if second_oder:
            start_idx = int(self.n_dn / 2)
            n_dn = int(self.n_dn / 2)  # we have n_dn/2 + 1 states (disp, vel, acc)
            derivatives = ["", "_dt", "_ddt"]
        else:
            start_idx = 0
            n_dn = self.n_dn
            derivatives = ["", "_dt"]

        self.save_traj_as_csv(
            self.X[:, :, dof],
            self.X_dt[:, :, dof, start_idx:],
            n_dn=n_dn,
            path=path,
            derivatives=derivatives,
            filename=filename,
        )

    def save_traj_as_csv(
        self,
        state,
        state_dt,
        n_dn,
        path,
        state_var="x",
        derivatives=["", "_dt"],
        filename="state_trajectories",
    ):
        """
        Save state trajectories and their time derivatives to a CSV file.

        This method concatenates state trajectories with their time derivatives and saves them
        to a CSV file. The CSV file includes time steps, state variables, and their derivatives.
        The output file is saved in the specified directory with a filename that can be customized.

        Parameters:
        -----------
        state : np.ndarray
            Array of state trajectories with shape (n_sim, n_t, n_dn), where n_sim is the number
            of simulations, n_t is the number of time steps, and n_dn is the number of degrees of
            freedom per node.

        state_dt : np.ndarray
            Array of state time derivatives with shape (n_sim, n_t, n_dn), where n_sim, n_t,
            and n_dn are as defined above.

        n_dn : int
            Number of degrees of freedom per node.

        path : str
            Directory path where the CSV file will be saved.

        state_var : str, optional
            Prefix for the state variables in the CSV header. Default is "x".

        derivatives : list of str, optional
            List of strings representing the derivatives to include in the header. Default is
            ["", "_dt"] for state and its first time derivative.

        filename : str, optional
            Base name for the output CSV file. The file will be saved with this name and a ".csv"
            extension. Default is "state_trajectories".
        """

        x_save_states = np.concatenate(
            [
                state,
                state_dt,
            ],
            axis=2,
        ).transpose([1, 2, 0])

        x_save_states = np.concatenate(
            [self.t, x_save_states.reshape(len(self.t), -1)], axis=1
        )

        header = ",".join(
            ["t"]
            + [
                f"{state_var}_{i}{derivative}_{j}"
                for derivative in derivatives
                for i in range(n_dn)
                for j in range(self.n_sim)
            ]
        )

        np.savetxt(
            os.path.join(path, f"{filename}.csv"),
            x_save_states,
            delimiter=",",
            fmt="%s",
            comments="",
            header=header,
        )

    def reproject_with_basis(
        self,
        V: np.ndarray | list[np.ndarray],
        idx: slice | list[slice] | None = None,
        pick_method: str = "all",
        pick_entry: None | list[int] | int | np.ndarray = None,
        seed: None | int = None,
    ):
        """
        Reprojects the state and its derivatives onto a new basis defined by the provided matrices `V`.

        This method applies a projection of the state `x`, its time derivatives `dx_dt`, and optional reconstructed states
        onto a basis formed from the matrices in `V`. It can handle multiple basis matrices and adjust the basis size
        based on the provided slices (`idx`) or the picking method (`pick_method`). The projection is done for the
        states and their time derivatives, with the option to project reconstructed states as well.

        Parameters:
        -----------
        V : np.ndarray or list of np.ndarray
            A matrix or list of matrices defining the basis for the projection. Each matrix corresponds to a basis
            that will be applied to the state and its time derivatives.

        idx : slice or list of slice, optional
            A slice or list of slices that defines the range of the matrices from `V` to use for projection. If not provided,
            it defaults to the entire range.

        pick_method : str, optional
            The method used for selecting which entries of `V` to pick. Options are:
            - "all" to pick all entries.
            - "rand" to randomly select entries.
            - "idx" to use specific indices from `pick_entry`.

        pick_entry : None, list[int], int, or np.ndarray, optional
            A list of indices or a single integer specifying which entries of `V` to pick. This is used when `pick_method`
            is set to "rand" or "idx". If `None`, it will default to picking all entries.

        seed : None or int, optional
            A random seed for reproducibility. Used when `pick_method` is "rand" to ensure the same indices are chosen
            across different runs.

        Returns:
        --------
        None
            This method does not return any value. It updates the internal state with the reprojected states (`x`),
            derivatives (`dx_dt`), and optionally reconstructed states (`x_rec`, `x_rec_dt`).

        Notes:
        ------
        - The projection is applied to the state `x`, its time derivatives `dx_dt`, and any reconstructed states (`x_rec`,
        `x_rec_dt`) if they exist.
        - The new basis for the state vector `x` and its derivatives is constructed from the provided matrices `V` and selected
        entries from `pick_list`.
        - The `features_to_states()` method is called at the end to convert the reprojected feature representation back to state
        space.
        """
        if self.x is None:
            self.states_to_features()
        if idx is None:
            idx = slice(self.n_f)
        pick_list = self.choose_picking_entries(V, pick_method, pick_entry, seed=seed)
        self.x, V_overall = self.reproject_x_with_basis(
            V=V, x=self.x, idx=idx, pick_list=pick_list
        )
        self.dx_dt, _ = self.reproject_x_with_basis(
            V=V, x=self.dx_dt, idx=idx, pick_list=pick_list
        )
        self.n_dn = V_overall.shape[0]
        if hasattr(self, "x_rec"):
            self.x_rec, _ = self.reproject_x_with_basis(
                V=V, x=self.x_rec, idx=idx, pick_list=pick_list
            )
            self.X_rec = reshape_features_to_states(
                self.x_rec, self.n_sim, self.n_t, n_n=self.n_n, n_dn=self.n_dn
            )
        if hasattr(self, "x_rec_dt"):
            self.x_rec_dt, _ = self.reproject_x_with_basis(
                V=V, x=self.x_rec_dt, idx=idx, pick_list=pick_list
            )
            self.X_rec_dt = reshape_features_to_states(
                self.x_rec_dt, self.n_sim, self.n_t, n_n=self.n_n, n_dn=self.n_dn
            )

        self.features_to_states()

    @staticmethod
    def choose_picking_entries(
        V: np.ndarray | list[np.ndarray],
        pick_method: str = "all",
        pick_entry: None | list[int] | int | np.ndarray = None,
        seed: None | int = None,
    ):
        """
        Select entries from the input array(s) based on the specified picking method.

        This method allows for selecting specific entries from a given array or list of arrays (`V`) using one of three
        picking methods: selecting all entries, randomly selecting entries, or using specified indices.

        Parameters:
        -----------
        V : np.ndarray or list of np.ndarray
            The input array or list of arrays from which entries will be selected. Each array is expected to have at least
            one dimension.

        pick_method : str, optional
            The method to use for selecting entries. It can be one of the following:
            - 'all': Selects all entries from each array in `V` (default).
            - 'rand': Selects a random subset of entries, with the number specified by `pick_entry`.
            - 'idx': Selects entries based on specific indices provided in `pick_entry`.

        pick_entry : int, list of int, np.ndarray, optional
            The number of entries to randomly select (`pick_method='rand'`), or the specific indices to select (`pick_method='idx'`).
            If `pick_method='rand'`, this should be an integer specifying the number of random entries to pick.
            If `pick_method='idx'`, this should be a list, numpy array, or integer specifying the indices to select.

        seed : int, optional
            A random seed to ensure reproducibility when using the 'rand' method for picking random entries. If not provided,
            the selection may vary each time.

        Returns:
        --------
        list of np.ndarray
            A list of arrays where each array contains the indices of the selected entries for each array in `V`.

        Notes:
        ------
        - If `V` is a list of arrays, each array is treated separately, and entries are selected from each array individually.
        - If `pick_method='rand'` and no seed is provided, a warning is logged indicating that the results may not be reproducible.
        - The selected entries are returned as a list of arrays corresponding to each input array in `V`.
        """
        assert pick_method in ["all", "rand", "idx"]
        if pick_method == "rand" and seed is None and isinstance(V, list):
            logging.warning(
                f"In order to get the same indices for each V a seed parameter is required."
            )
        if not isinstance(V, list):
            V = [V]
        pick_list = []
        for V_temp in V:
            if pick_method == "all":
                pick_list.append(np.arange(V_temp.shape[0]))
            elif pick_method == "rand":
                assert isinstance(pick_entry, int)
                rng = np.random.default_rng(seed=seed)
                idx_rand = sorted(
                    rng.choice(
                        V_temp.shape[0],
                        size=(pick_entry,),
                        replace=False,
                    )
                )
                pick_list.append(idx_rand)
            elif pick_method == "idx":
                assert (
                    isinstance(pick_entry, list)
                    or isinstance(pick_entry, np.ndarray)
                    or isinstance(pick_entry, int)
                )
                pick_list.append(pick_entry)
        return pick_list

    @staticmethod
    def reproject_x_with_basis(
        V: np.ndarray | list[np.ndarray],
        x: np.ndarray,
        idx: slice | list[slice] | None = None,
        pick_list: np.ndarray | list[np.ndarray] | None = None,
    ):
        """
        Reprojects the state vector `x` onto a new basis constructed from the provided matrices `V`.

        This method constructs an overall basis by combining identity matrices and the selected submatrices from `V`,
        then reprojects the state vector `x` onto this new basis. This projection is useful in various machine learning
        and optimization tasks where a lower-dimensional representation of the data is required.

        Parameters:
        -----------
        V : np.ndarray or list of np.ndarray
            The set of basis matrices to use for the projection. If multiple matrices are provided, they are used in
            combination to form the overall basis.

        x : np.ndarray
            The state vector that will be reprojected. It should have the shape (n_t, n_f), where `n_t` is the number of
            time steps and `n_f` is the number of features.

        idx : slice or list of slice, optional
            A slice or list of slices that define the specific submatrices of `V` to use in constructing the overall basis.
            If `V` is a list of matrices, `idx` should also be a list of slices, corresponding to each matrix in `V`.

        pick_list : np.ndarray or list of np.ndarray, optional
            A list of indices specifying which entries to pick from each matrix in `V`. This is used when the size of `V`
            is too large for a full calculation, allowing for selective picking of rows.

        Returns:
        --------
        xT : np.ndarray
            The reprojected state vector, with the same shape as `x`.

        V_overall : np.ndarray
            The overall basis matrix that was used to reproject `x`. This matrix is the block diagonal matrix formed
            by combining identity matrices and the selected submatrices from `V`.

        Notes:
        ------
        - The `pick_list` is only necessary when `V.shape[0]` is too large for a full calculation and needs to be sampled.
        - If `idx` and `V` are lists, each slice in `idx` corresponds to a submatrix in `V`, and the corresponding picking
        indices from `pick_list` will be used for each submatrix.
        - The `block_diag` function is used to form the overall basis matrix by creating a block diagonal matrix from the
        individual identity matrices and submatrices selected from `V`.
        """
        n_f = x.shape[1]

        if isinstance(V, list):
            assert isinstance(idx, list)
            start = 0
            for slicer in idx:
                # assert slicer are in correct order
                assert slicer.start >= start
                start = slicer.stop
        else:
            V = [V]
            idx = [idx]

        # %% create overall basis
        arrays = []
        slicer_stop = 0  # initialize
        for i, slicer in enumerate(idx):
            arrays.append(np.eye(slicer.start - slicer_stop))
            if pick_list is None:
                pick_temp = np.arange(V[i].shape[0])
            else:
                if isinstance(pick_list, list):
                    pick_temp = pick_list[i]
                else:
                    pick_temp = pick_list
            V_pick = V[i][pick_temp, :]
            arrays.append(V_pick)
            slicer_stop = slicer.stop
        arrays.append(np.eye(n_f - slicer_stop))
        V_overall = block_diag(*arrays)
        xT = V_overall @ x.T
        return xT.T, V_overall


class PHIdentifiedData(Data):
    """
    Class representing the identified port-Hamiltonian data which was obtained through the aphin framework.

    """

    def __init__(
        self,
        t,
        X,
        X_dt=None,
        U=None,
        Mu=None,
        x_ph=None,
        dx_dt_ph=None,
        z=None,
        Z=None,
        z_dt=None,
        Z_dt=None,
        x_rec=None,
        X_rec=None,
        x_rec_dt=None,
        X_rec_dt=None,
        z_dt_ph_map=None,
        Z_dt_ph_map=None,
        z_ph=None,
        Z_ph=None,
        z_dt_ph=None,
        Z_dt_ph=None,
        H_ph=None,
        n_red=None,
        J=None,
        R=None,
        B=None,
        Q=None,
        solving_times=None,
        is_scaled=False,
        scaling_values=None,
        **kwargs,
    ):
        """
        Initializes a PHIdentifiedData instance.

        Parameters:
        -----------
        t : np.ndarray
            Time steps array with shape (n_t,).

        X : np.ndarray
            State data array with shape (n_sim, n_t, n_n, n_dn), where:
            - n_sim: number of simulations
            - n_t: number of time steps
            - n_n: number of nodes
            - n_dn: number of degrees of freedom per node

        X_dt : np.ndarray, optional
            Time derivative of state data with shape (n_sim, n_t, n_n, n_dn). Default is None.

        U : np.ndarray, optional
            Input data array with shape (n_sim, n_t, n_u), where n_u is the number of inputs. Default is None.

        Mu : np.ndarray, optional
            Parameter data with shape (n_sim, n_mu), where n_mu is the number of parameters. Default is None.

        x_ph : np.ndarray, optional
            Parameterized Port-Hamiltonian state data with shape (n_sim * n_t, n_n * n_dn). Default is None.

        dx_dt_ph : np.ndarray, optional
            Time derivative of parameterized Port-Hamiltonian  state data with shape (n_sim * n_t, n_n * n_dn). Default is None.

        z : np.ndarray, optional
            Latent data with shape (n_sim * n_t, n_z), where n_z is the number of latent variables. Default is None.

        Z : np.ndarray, optional
            Parameterized Port-Hamiltonian latent data with shape (n_sim, n_t, n_z). Default is None.

        z_dt : np.ndarray, optional
            Time derivative of latent data with shape (n_sim * n_t, n_z). Default is None.

        Z_dt : np.ndarray, optional
            Time derivative of parameterized Port-Hamiltonian latent data with shape (n_sim * n_t, n_z). Default is None.

        x_rec : np.ndarray, optional
            Reconstructed state data with shape (n_sim* n_t, n_n * n_dn). Default is None.

        X_rec : np.ndarray, optional
            Reconstructed state data array with shape (n_sim, n_t, n_n, n_dn). Default is None.

        x_rec_dt : np.ndarray, optional
            Time derivative of reconstructed state data with shape (n_sim * n_t, n_n * n_dn). Default is None.

        X_rec_dt : np.ndarray, optional
            Time derivative of reconstructed state data array with shape (n_sim, n_t, n_n, n_dn). Default is None.

        z_dt_ph_map : np.ndarray, optional
            Mapping of time derivatives for latent data in the parameterized Port-Hamiltonian with shape (n_sim, n_t, n_z). Default is None.

        Z_dt_ph_map : np.ndarray, optional
            Mapping of time derivatives for parameterized Port-Hamiltonian  latent data with shape (n_sim, n_t, n_z). Default is None.

        z_ph : np.ndarray, optional
            Parameterized Port-Hamiltonian  latent data with shape (n_sim * n_t, n_z). Default is None.

        Z_ph : np.ndarray, optional
            Parameterized Port-Hamiltonian  latent data with shape (n_sim, n_t, n_z). Default is None.

        z_dt_ph : np.ndarray, optional
            Time derivative of parameterized Port-Hamiltonian latent data with shape (n_sim * n_t, n_z). Default is None.

        Z_dt_ph : np.ndarray, optional
            Time derivative of parameterized Port-Hamiltonian latent data with shape (n_sim, n_t, n_z). Default is None.

        H_ph : np.ndarray, optional
            Port-Hamiltonian function data with shape (n_sim, n_t, ...). Default is None.

        n_red : int, optional
            Number of reduced order parameters. Default is None.

        J : np.ndarray, optional
            Matrix J with shape (n_sim, n_t, ...). Default is None.

        R : np.ndarray, optional
            Matrix R with shape (n_sim, n_t, ...). Default is None.

        B : np.ndarray, optional
            Matrix B with shape (n_sim, n_t, ...). Default is None.

        Q : np.ndarray, optional
            Matrix Q with shape (n_sim, n_t, ...). Default is None.

        solving_times : np.ndarray, optional
            Times associated with solving the system, with shape (n_t,). Default is None.

        is_scaled : bool, optional
            A flag indicating whether the data has been scaled. Default is False.

        scaling_values : np.ndarray, optional
            Scaling factors to be applied to the data, used when `is_scaled` is True. Default is None.

        **kwargs : additional keyword arguments
            Additional parameters passed to the parent class `Data`.

        Notes:
        ------
        This class is designed for handling Port-Hamiltonian systems, which are characterized by their
        specific structure and conservation properties. It includes attributes and methods for managing
        both state and auxiliary data, as well as for working with the parameterized hypotheses and
        reconstructed data.
        """
        super().__init__(t, X, X_dt, U=U, Mu=Mu, J=J, R=R, B=B, Q=Q)
        if n_red is None:
            if z is not None:
                self.n_red = z.shape[1]
            elif Z is not None:
                self.n_red = Z.shape[2]
            else:
                self.n_red = n_red

        else:
            self.n_red = n_red
        self.x = x_ph
        self.dx_dt = dx_dt_ph
        self.z = z
        self.Z = Z
        self.z_dt = z_dt
        self.Z_dt = Z_dt
        self.x_rec = x_rec
        self.X_rec = X_rec
        self.x_rec_dt = x_rec_dt
        self.X_rec_dt = X_rec_dt
        self.z_dt_ph_map = z_dt_ph_map
        self.Z_dt_ph_map = Z_dt_ph_map
        self.z_ph = z_ph
        self.Z_ph = Z_ph
        self.z_dt_ph = z_dt_ph
        self.Z_dt_ph = Z_dt_ph
        self.H_ph = H_ph
        self.solving_times = solving_times
        self.is_scaled = is_scaled
        self.scaling_values = scaling_values

    @staticmethod
    def obtain_results_from_ph_autoencoder(data, system_layer, ph_network):
        """
        Obtain relevant results from the identified port-Hamiltonian autoencoder (APHIN), including latent variables,
        time derivatives, and reconstructed states.

        This method extracts and computes the following from the port-Hamiltonian autoencoder:
        - Latent variables and their time derivatives.
        - Reconstructed states and their time derivatives.
        It also reshapes the feature arrays into state arrays suitable for further analysis.

        Parameters
        ----------
        data : Data
            Dataset object containing the input data used for encoding and reconstructing.
        system_layer : SystemLayer
            System layer object defining the system's latent space dimensionality and configuration.
        ph_network : APHIN
            Port-Hamiltonian autoencoder object used for encoding, reconstructing, and computing time derivatives.

        Returns
        -------
        tuple
            A tuple containing:
            - z : numpy.ndarray
                Latent variables obtained from encoding the input data.
            - z_dt : numpy.ndarray
                Time derivatives of the latent variables.
            - x_rec : numpy.ndarray
                Reconstructed states from the latent variables.
            - x_rec_dt : numpy.ndarray
                Time derivatives of the reconstructed states.
            - Z : numpy.ndarray
                Reshaped latent variables into state array format.
            - Z_dt : numpy.ndarray
                Reshaped time derivatives of the latent variables into state array format.
            - X_rec : numpy.ndarray
                Reshaped reconstructed states into state array format.
            - X_rec_dt : numpy.ndarray
                Reshaped time derivatives of the reconstructed states into state array format.
        """
        # calculate latent variables from data
        z = ph_network.encode(data.x).numpy()

        # calculate time derivative of latent variable through automatic differentiation from data
        _, z_dt = ph_network.calc_latent_time_derivatives(data.x, data.dx_dt)

        # reconstruct states from data
        x_rec = ph_network.reconstruct(data.x).numpy()

        # calculate time derivative of state variable through automatic differentiation from data
        _, x_rec_dt = ph_network.calc_physical_time_derivatives(z, z_dt)

        # reshape feature arrays to state arrays
        n_sim, n_t = data.n_sim, data.n_t
        n_f, n_n, n_dn = system_layer.r, data.n_n, data.n_dn
        state_arrays = dict(
            Z=dict(x=z, n_sim=n_sim, n_t=n_t, n_f=n_f),
            Z_dt=dict(x=z_dt, n_sim=n_sim, n_t=n_t, n_f=n_f),
            X_rec=dict(x=x_rec, n_sim=n_sim, n_t=n_t, n_n=n_n, n_dn=n_dn),
            X_rec_dt=dict(x=x_rec_dt, n_sim=n_sim, n_t=n_t, n_n=n_n, n_dn=n_dn),
        )
        for key, val in state_arrays.items():
            state_arrays[key] = reshape_features_to_states(**val)

        return (
            z,
            z_dt,
            x_rec,
            x_rec_dt,
            state_arrays["Z"],
            state_arrays["Z_dt"],
            state_arrays["X_rec"],
            state_arrays["X_rec_dt"],
        )

    @staticmethod
    def obtain_results_from_ph_network(data, system_layer):
        """
        Obtain relevant results from the identified pH network, including state variables and time derivatives.

        This method extracts and reshapes the state variables and their time derivatives from the provided
        dataset. It also prepares placeholders for the reconstructed states and their time derivatives
        to ensure conformity with the expected output format.

        Parameters
        ----------
        data : Data
            Dataset object containing the input data and time derivatives.
        system_layer : SystemLayer
            System layer object defining the latent space dimensionality.

        Returns
        -------
        tuple
            A tuple containing:
            - z : numpy.ndarray
                Latent variables (high-dimensional states).
            - z_dt : numpy.ndarray
                Time derivatives of the latent variables.
            - x_rec : numpy.ndarray
                Reconstructed states (if applicable, otherwise None).
            - x_rec_dt : numpy.ndarray
                Time derivatives of the reconstructed states (if applicable, otherwise None).
            - Z : numpy.ndarray
                Reshaped latent variables into state array format.
            - Z_dt : numpy.ndarray
                Reshaped time derivatives of the latent variables into state array format.
            - X_rec : numpy.ndarray
                Reshaped reconstructed states into state array format (if applicable, otherwise None).
            - X_rec_dt : numpy.ndarray
                Reshaped time derivatives of the reconstructed states into state array format (if applicable, otherwise None).
        """
        # no autoencoder: z = x
        z = data.x
        Z = reshape_features_to_states(z, data.n_sim, data.n_t, n_f=system_layer.r)
        z_dt = data.dx_dt
        Z_dt = reshape_features_to_states(
            z_dt, data.n_sim, data.n_t, n_f=system_layer.r
        )
        (
            x_rec,
            X_rec,
            x_rec_dt,
            X_rec_dt,
        ) = [None] * 4
        return (
            z,
            z_dt,
            x_rec,
            x_rec_dt,
            Z,
            Z_dt,
            X_rec,
            X_rec_dt,
        )

    @staticmethod
    def obtain_ph_map_data(ph_network, z, data, n_f):
        """
        Obtain time derivatives of latent variables using the pH network and reshape them into state format.

        This method calculates the time derivatives of latent variables using the pH network and reshapes
        them into a format consistent with the state arrays.

        Parameters
        ----------
        ph_network : PHNetwork
            pH network object used to compute time derivatives.
        z : numpy.ndarray
            Latent variables for which the time derivatives are to be computed.
        data : Data
            Dataset object containing input data and parameters.
        n_f : int
            Number of latent variables (features) used in reshaping.

        Returns
        -------
        tuple
            A tuple containing:
            - z_dt_ph_map : numpy.ndarray
                Time derivatives of the latent variables as computed by the pH network.
            - Z_dt_ph_map : numpy.ndarray
                Reshaped time derivatives of the latent variables into state array format.
        """
        if data.mu is None:
            mu_input = np.expand_dims(np.array([]), axis=0)
        else:
            mu_input = data.mu
        if data.u is None:
            u_input = np.expand_dims(np.array([]), axis=0)
        else:
            u_input = data.u
        z_dt_ph_map = ph_network.system_network([z, u_input, mu_input]).numpy()
        Z_dt_ph_map = reshape_features_to_states(
            z_dt_ph_map,
            data.n_sim,
            data.n_t,
            n_f=n_f,
        )
        return z_dt_ph_map, Z_dt_ph_map

    @staticmethod
    def obtain_ph_data(
        data,
        ph_network,
        system_layer,
        J_ph,
        R_ph,
        B_ph,
        Q_ph,
        E_ph,
        integrator_type,
        decomp_option,
        calc_u_midpoints=False,
    ):
        """
        Obtain the port-Hamiltonian (pH) system data, including latent variables, time derivatives,
        reconstructed states, and Hamiltonian values.

        This method computes the reduced trajectories for the identified pH system based on the provided
        data, pH network, and system parameters. It calculates latent variables and their time derivatives,
        reconstructs states and their time derivatives, and computes the Hamiltonian for each simulation.

        Parameters
        ----------
        data : Data
            Dataset object containing the input data and initial conditions.
        ph_network : APHIN or PHIN
            Port-Hamiltonian autoencoder object used for encoding, reconstructing, and computing derivatives.
        system_layer : SystemLayer
            System layer object defining the latent space dimensionality and configuration.
        J_ph : numpy.ndarray
            System matrix J for the pH system.
        R_ph : numpy.ndarray
            System matrix R for the pH system.
        B_ph : numpy.ndarray, optional
            System matrix B for the pH system, by default None.
        Q_ph : numpy.ndarray, optional
            System matrix Q for the pH system, by default None.
        E_ph : numpy.ndarray, optional
            System matrix E for the pH system, by default None.
        integrator_type : str
            Type of integrator used for solving the pH system.
        decomp_option : str
            Decomposition option for solving the pH system.
        calc_u_midpoints : bool, optional
            If True, calculates the midpoints of the input data (U) for numerical integration.
            If False, uses the input data as is. Default is False.

        Returns
        -------
        tuple
            A tuple containing:
            - z_ph : numpy.ndarray
                Latent variables for each simulation.
            - dz_dt_ph : numpy.ndarray
                Time derivatives of the latent variables.
            - x_ph : numpy.ndarray
                Reconstructed states for each simulation.
            - dx_dt_ph : numpy.ndarray
                Time derivatives of the reconstructed states.
            - Z_ph : numpy.ndarray
                Reshaped latent variables into state array format.
            - Z_dt_ph : numpy.ndarray
                Reshaped time derivatives of the latent variables into state array format.
            - X_ph : numpy.ndarray
                Reshaped reconstructed states into state array format.
            - X_dt_ph : numpy.ndarray
                Reshaped time derivatives of the reconstructed states into state array format.
            - H_ph : numpy.ndarray
                Hamiltonian values for each simulation.
        """
        # initialize arrays
        system_ph_list = []
        latent_shape = (data.n_sim, data.n_t, system_layer.r)
        state_shape = (data.n_sim, data.n_t, data.n_n, data.n_dn)
        Z_ph = np.zeros(latent_shape)
        Z_dt_ph = np.zeros(latent_shape)
        X_ph = np.zeros(state_shape)
        X_dt_ph = np.zeros(state_shape)
        H_ph = np.zeros((data.n_sim, data.n_t))

        logging.info(f"Calculating reduced trajectories of the identified system.")
        solving_times = []
        for i_sim in tqdm(range(data.n_sim)):
            x_init = np.expand_dims(
                data.x_init[i_sim, :], axis=0
            )  # tf somehow expects (None,n_f)
            if data.mu is None:
                if E_ph is None:
                    system_ph = PHSystem(J_ph, R_ph, B=B_ph, Q_ph=Q_ph)
                else:
                    system_ph = DescrPHSystem(
                        J_ph,
                        R_ph,
                        E_ph=E_ph,
                        B=B_ph,
                        Q_ph=Q_ph,
                    )
            else:
                if E_ph is None:
                    system_ph = PHSystem(
                        J_ph[i_sim],
                        R_ph[i_sim],
                        B=B_ph[i_sim] if B_ph is not None else None,
                        Q_ph=Q_ph[i_sim] if Q_ph is not None else None,
                    )
                else:
                    system_ph = DescrPHSystem(
                        J_ph[i_sim],
                        R_ph[i_sim],
                        E_ph=E_ph[i_sim],
                        B=B_ph[i_sim] if B_ph is not None else None,
                        Q_ph=Q_ph[i_sim] if Q_ph is not None else None,
                    )

            system_ph_list.append(system_ph)

            start_time = time.time()

            # ENCODING
            if isinstance(ph_network, APHIN):
                z_init = ph_network.encode(x_init).numpy().T
            elif isinstance(ph_network, PHIN):
                z_init = x_init.T

            # SOLVE LATENT DYNAMICS
            if data.U is None:
                u = None
            else:
                if calc_u_midpoints:
                    u = np.zeros_like(data.U[i_sim])
                    u[:-1] = (
                        data.U[i_sim, 1:] + data.U[i_sim, :-1]
                    ) / 2  # last u step is not used, padded for input format
                else:
                    u = data.U[i_sim]
            Z_ph[i_sim], Z_dt_ph[i_sim] = system_ph_list[i_sim].solve_dt(
                data.t,
                z_init,
                u,
                integrator_type=integrator_type,
                decomp_option=decomp_option,
            )
            # Reshape to (n_t * n_s, r)
            z_ph, dz_dt_ph = reshape_states_to_features(
                np.expand_dims(Z_ph[i_sim], axis=0),
                np.expand_dims(Z_dt_ph[i_sim], axis=0),
            )

            # z_ref = ph_network.encode(data.x[i_sim*1001:(i_sim+1)*1001]).numpy()
            # import matplotlib.pyplot as plt
            # i_state = 6
            # plt.figure()
            # plt.plot(Z_ph[i_sim, :, i_state])
            # plt.plot(z_ref[:, i_state])
            # plt.show()

            # DECODING
            if isinstance(ph_network, APHIN):
                # decode time integrated latent state
                x_ph = ph_network.decoder(z_ph).numpy()
                x_ph_, dx_dt_ph = ph_network.calc_physical_time_derivatives(
                    z_ph, dz_dt_ph
                )
            elif isinstance(ph_network, PHIN):
                # no autoencoder: x = z
                x_ph = z_ph
                dx_dt_ph = dz_dt_ph

            end_time = time.time()
            solving_times.append(end_time - start_time)

            X_ph[i_sim], X_dt_ph[i_sim] = reshape_features_to_states(
                x_ph,
                1,
                data.n_t,
                x_dt=dx_dt_ph,
                n_n=data.n_n,
                n_dn=data.n_dn,
            )
            # calculate Hamiltonian
            H_ph[i_sim, :] = system_ph_list[i_sim].H(Z_ph[i_sim, :, :])

        solving_times = dict(
            per_run=np.array(solving_times), mean=np.mean(solving_times)
        )
        logging.info(
            f"Average time for solving the system for one simulation: {solving_times['mean']:.5f} seconds."
        )

        # combine all simulations to one array
        z_ph, dz_dt_ph = reshape_states_to_features(Z_ph, Z_dt_ph)
        x_ph, dx_dt_ph = reshape_states_to_features(X_ph, X_dt_ph)

        return (
            z_ph,
            dz_dt_ph,
            x_ph,
            dx_dt_ph,
            Z_ph,
            Z_dt_ph,
            X_ph,
            X_dt_ph,
            H_ph,
            solving_times,
        )

    def states_to_features(self):
        super().states_to_features()
        if self.Z_ph is not None:
            self.z_ph, self.z_dt_ph = reshape_states_to_features(
                self.Z_ph, self.Z_dt_ph
            )
        if self.Z is not None:
            self.z, self.z_dt = reshape_states_to_features(self.Z, self.Z_dt)
        if self.Z_dt_ph_map is not None:
            self.z_dt_ph_map = reshape_states_to_features(self.Z_dt_ph_map)

    @classmethod
    def from_identification(
        cls,
        data,
        system_layer,
        ph_network,
        integrator_type="IMR",
        decomp_option="lu",
        calc_u_midpoints=False,
        **kwargs,
    ):
        """
        Create an instance of PHIdentifiedData from the identified pH system.

        This method initializes an instance of the PHIdentifiedData class using results obtained from an identified
        port-Hamiltonian (pH) system. It calculates latent variables, their time derivatives, and reconstructed states,
        and performs simulation using the identified pH system. The method supports both APHIN and PHIN networks.

        Parameters
        ----------
        cls : type
            The class to instantiate.
        data : Data
            The dataset object containing the initial conditions and other data.
        system_layer : SystemLayer
            The system layer object used to extract system matrices.
        ph_network : APHIN or PHIN
            The pH network object used for encoding, reconstructing, and calculating time derivatives.
        integrator_type : str, optional
            The type of integrator used for simulation (default is "IMR").
        decomp_option : str, optional
            The decomposition option for solving the system (default is "lu").
        calc_u_midpoints : bool, optional
            If `True`, calculates the midpoints of the input data (`U`) for numerical integration. If `False`, uses the input data as is. Default is `False`.
        **kwargs : dict
            Additional keyword arguments to pass to the PHIdentifiedData class constructor.

        Returns
        -------
        PHIdentifiedData
            An instance of the PHIdentifiedData class initialized with results from the identified pH system.
        """
        logging.info(f"Obtaining relevant results from the identified system.")
        if isinstance(ph_network, APHIN):
            (
                z,
                z_dt,
                x_rec,
                x_rec_dt,
                Z,
                Z_dt,
                X_rec,
                X_rec_dt,
            ) = cls.obtain_results_from_ph_autoencoder(data, system_layer, ph_network)

        elif isinstance(ph_network, PHIN):
            (
                z,
                z_dt,
                x_rec,
                x_rec_dt,
                Z,
                Z_dt,
                X_rec,
                X_rec_dt,
            ) = cls.obtain_results_from_ph_network(data, system_layer)
        else:
            raise NotImplementedError(
                f"Type of phin instance {type(ph_network)} is not implemented yet."
            )

        # %% get time derivative of latent state through port-Hamiltonian mapping with the pH matrices (z,u,mu is given)
        z_dt_ph_map, Z_dt_ph_map = cls.obtain_ph_map_data(
            ph_network, z, data, system_layer.r
        )

        # %% calculate latent state coordinates, i.e. the pH state, through time integration
        if data.x_init is None:
            data.get_initial_conditions()

        # get system matrices
        J_ph, R_ph, B_ph, Q_ph, E_ph = Data.get_system_matrices_from_system_layer(
            system_layer=system_layer, mu=data.mu, n_t=data.n_t
        )

        if calc_u_midpoints:
            logging.info(f"Calculating midpoints from given input U.")
        # simulate training data
        (
            z_ph_s,
            dz_dt_ph_s,
            x_ph_s,
            dx_dt_ph_s,
            Z_ph,
            Z_dt_ph,
            X_ph,
            X_dt_ph,
            H_ph,
            solving_times,
        ) = cls.obtain_ph_data(
            data,
            ph_network,
            system_layer,
            J_ph,
            R_ph,
            B_ph,
            Q_ph,
            E_ph,
            integrator_type,
            decomp_option,
            calc_u_midpoints,
        )

        if data.is_scaled:
            is_scaled = True
            scaling_values = data.scaling_values
        else:
            scaling_values = None
            is_scaled = False

        return cls(
            t=data.t,
            X=X_ph,
            X_dt=X_dt_ph,
            U=data.U,
            Mu=data.Mu,
            x_ph=x_ph_s,
            dx_dt_ph=dx_dt_ph_s,
            z=z,
            Z=Z,
            z_dt=z_dt,
            Z_dt=Z_dt,
            x_rec=x_rec,
            X_rec=X_rec,
            x_rec_dt=x_rec_dt,
            X_rec_dt=X_rec_dt,
            z_dt_ph_map=z_dt_ph_map,
            Z_dt_ph_map=Z_dt_ph_map,
            z_ph=z_ph_s,
            Z_ph=Z_ph,
            z_dt_ph=dz_dt_ph_s,
            Z_dt_ph=Z_dt_ph,
            H_ph=H_ph,
            n_red=system_layer.r,
            solving_times=solving_times,
            is_scaled=is_scaled,
            scaling_values=scaling_values,
            **kwargs,
        )

    @classmethod
    def from_system_list(cls, system_list: list, data: Data):
        """
        Create an instance of PHIdentifiedData from a list of identified systems.

        This method generates a PHIdentifiedData instance using a list of identified systems and a dataset.
        It processes the given system list and computes the relevant state variables and system matrices, including
        latent states, time derivatives, and the system matrices (J, R, B, Q). The method assumes that the system list
        is compatible with the PHIN case.

        Parameters
        ----------
        cls : type
            The class to instantiate.
        system_list : list
            A list of identified pH systems, where each system corresponds to a simulation.
        data : Data
            The dataset object containing the initial conditions, system input, and other data.

        Returns
        -------
        PHIdentifiedData
            An instance of the PHIdentifiedData class populated with computed states and matrices from the system list.

        Notes
        -----
        This method is currently implemented for the PHIN case only.
        """
        logging.info(
            f"PHIdentifiedData.from_system_list has only been implemented for the PHIN case."
        )

        n_f = system_list[0].n
        latent_shape = (
            data.n_sim,
            data.n_t,
            n_f,
        )
        assert len(system_list) == data.n_sim

        # initialize solution arrays
        Z_ph = np.zeros(latent_shape)
        Z_dt_ph = np.zeros(latent_shape)

        # initialize matrices
        J = np.zeros((data.n_sim, n_f, n_f))
        R = np.zeros((data.n_sim, n_f, n_f))
        Q = np.zeros((data.n_sim, n_f, n_f))
        B = np.zeros((data.n_sim, n_f, data.n_u))

        data.get_initial_conditions()
        for i_sim in range(data.n_sim):
            u = data.U[i_sim]
            x_init = np.expand_dims(data.x_init[i_sim, :], axis=0).T
            Z_ph[i_sim], Z_dt_ph[i_sim] = system_list[i_sim].solve_dt(
                data.t,
                x_init,
                u,
            )
            J[i_sim], R[i_sim], B[i_sim], Q[i_sim] = system_list[
                i_sim
            ].get_system_matrix()

        z_ph, dz_dt_ph = reshape_states_to_features(Z_ph, Z_dt_ph)
        x_ph, dx_dt_ph = z_ph, dz_dt_ph
        X_ph, X_dt_ph = reshape_features_to_states(
            x_ph,
            data.n_sim,
            data.n_t,
            x_dt=dx_dt_ph,
            n_n=data.n_n,
            n_dn=data.n_dn,
        )
        z = data.x
        Z = reshape_features_to_states(
            z,
            data.n_sim,
            data.n_t,
            n_f=n_f,
        )
        z_dt = data.dx_dt
        Z_dt = reshape_features_to_states(
            z_dt,
            data.n_sim,
            data.n_t,
            n_f=n_f,
        )

        ph_identified_data_instance = cls(
            t=data.t,
            X=X_ph,
            X_dt=X_dt_ph,
            Z=Z,
            Z_dt=Z_dt,
            Z_ph=Z_ph,
            Z_dt_ph=Z_dt_ph,
            Mu=data.Mu,
            J=J,
            R=R,
            B=B,
            Q=Q,
        )
        ph_identified_data_instance.states_to_features()
        return ph_identified_data_instance

    def save_latent_traj_as_csv(self, path, filename="latent_trajectories"):
        """
        Save latent trajectories to a CSV file with the following format:
        t, z_0_isim, z_1_isim, ..., z_{r-1}_isim, z_0_dt_isim, z_1_dt_isim, ..., z_{r-1}_dt_isim

        This method saves both the predicted latent trajectories (`Z_ph`) and their time derivatives (`Z_dt_ph`)
        as well as the reference latent trajectories (`Z`) and their time derivatives (`Z_dt`) to CSV files.
        Each file includes the time vector `t`, followed by the latent variables and their time derivatives.

        Parameters
        ----------
        path : str
            The directory path where the CSV files will be saved.
        filename : str, optional
            The base filename for the CSV files. The default is "latent_trajectories". Additional suffixes
            will be added to differentiate between predicted and reference trajectories.
        """

        self.save_traj_as_csv(
            self.Z_ph, self.Z_dt_ph, self.n_red, path, state_var="z", filename=filename
        )

        self.save_traj_as_csv(
            self.Z,
            self.Z_dt,
            self.n_red,
            path,
            state_var="z",
            filename=f"{filename}_reference",
        )


class LTIDataset(Data):
    """
    Dataset class for Linear Time-Invariant (LTI) systems.

    Inherits from the `Data` class and is designed to handle datasets specifically for LTI systems.
    This class extends the functionality of the base `Data` class by including data that might be used
    for LTI system identification or analysis.
    """

    def __init__(
        self,
        t,
        X,
        U=None,
        X_dt=None,
    ):
        """
        Initialize the LTIDataset instance.

        Parameters
        ----------
        t : np.ndarray
            Time vector for the dataset.
        X : np.ndarray
            State trajectories.
        U : np.ndarray, optional
            Input trajectories, by default None.
        X_dt : np.ndarray, optional
            Time derivatives of the state trajectories, by default None.
        """
        super().__init__(t, X, X_dt, U)
        # might be used for other purposes
