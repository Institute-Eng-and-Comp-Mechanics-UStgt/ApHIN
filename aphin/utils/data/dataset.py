"""
Encapsulate data loading and data generation
"""

import numpy as np
import logging
import pandas as pd
import os
import re
from natsort import natsorted
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt


# own package
from aphin.utils.data import Data, PHIdentifiedData
from aphin.utils.transformations import reshape_inputs_to_features


class Dataset(Data):
    """
    Container for multiple datasets (train and test) with states, inputs, and parameters.

    Parameters:
    -----------
    t : ndarray
        Array of time steps.
    X : ndarray
        States array with shape (n_sim, n_t, n_n, n_dn).
    X_dt : ndarray
        Time derivatives of the states, with the same shape as X.
    U : ndarray, optional
        Input array with shape (n_sim, n_t, n_u). Default is None.
    Mu : ndarray, optional
        Parameters array with shape (n_sim, n_mu). Default is None.
    J : ndarray, optional
        pH interconnection matrix with shape (r, r, n_sim). Default is None.
    R : ndarray, optional
        pH dissipation matrix with shape (r, r, n_sim). Default is None.
    Q : ndarray, optional
        pH energy matrix with shape (r, r, n_sim). Default is None.
    B : ndarray, optional
        pH input matrix with shape (r, n_u, n_sim). Default is None.
    Mu_input : ndarray, optional
        Parameters that where defined for the creation of the input array U. Default is None.

    Attributes:
    -----------
    TRAIN : Dataset or None
        Training dataset. Initialized to None.
    TEST : Dataset or None
        Testing dataset. Initialized to None.

    Notes:
    ------
    - Inherits from the `Data` class, which handles the initialization and validation of the time steps,
      states, inputs, and parameters.
    - This class is designed to facilitate the handling of multiple datasets, including training and testing datasets.
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
        super().__init__(t, X, X_dt, U, Mu, J, R, Q, B, Mu_input)
        # initialize train and test data objects
        self.TRAIN = None
        self.TEST = None

    @property
    def data(self):
        """
        Get training data.

        Returns
        -------
        Data
            The training dataset object. This property returns the data associated
            with the training dataset, which is accessible through `self.TRAIN.data`.
        """
        return self.TRAIN.data

    @property
    def test_data(self):
        """
        Get test data.

        Returns
        -------
        Data
            The testing dataset object. This property returns the data associated
            with the testing dataset, which is accessible through `self.TEST.data`.
        """
        return self.TEST.data

    @property
    def ph_matrices(self):
        """
        Get port-Hamiltonian matrices from the training dataset.

        Returns
        -------
        tuple
            A tuple containing the port-Hamiltonian matrices associated with
            the training dataset, accessible through `self.TRAIN.ph_matrices`.
        """
        return self.TRAIN.ph_matrices

    @property
    def ph_matrices_test(self):
        """
        Get port-Hamiltonian matrices from the testing dataset.

        Returns
        -------
        tuple
            A tuple containing the port-Hamiltonian matrices associated with
            the testing dataset, accessible through `self.TEST.ph_matrices`.
        """
        return self.TEST.ph_matrices

    @property
    def Data(self):
        """
        Retrieve the state and derivative data from the container for the training dataset.

        Returns
        -------
        tuple
            A tuple containing:
            - X: States array with shape (n_sim, n_t, n_n, n_dn).
            - X_dt: Time derivatives of the states, with the same shape as X.
            - U: Input array with shape (n_sim, n_t, n_u), if available.
            - Mu: Parameters array with shape (n_sim, n_mu), if available.
        """
        return self.TRAIN.Data

    @property
    def Data_test(self):
        """
        Retrieve the state and derivative data from the test dataset.

        Returns
        -------
        tuple
            A tuple containing:
            - X: States array with shape (n_sim, n_t, n_n, n_dn) from the test dataset.
            - X_dt: Time derivatives of the states, with the same shape as X, from the test dataset.
            - U: Input array with shape (n_sim, n_t, n_u) from the test dataset, if available.
            - Mu: Parameters array with shape (n_sim, n_mu) from the test dataset, if available.
        """
        return self.TEST.Data

    @property
    def shape(self):
        """
        Return the shape of the training dataset.

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
        return self.TRAIN.shape

    @property
    def shape_test(self):
        """
        Return the shape of the testing dataset.

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
        return self.TEST.shape

    def permute_matrices(self, permutation_idx: list[int] | slice):
        """
        Permute the matrices of the training and testing datasets using the given permutation indices.

        This method applies the given permutation to the matrices in both the training and testing datasets.
        It modifies the states, time derivatives, inputs, and other relevant matrices based on the provided
        permutation indices or slice.

        Parameters
        ----------
        permutation_idx : list[int] | slice
            The indices or slice used for permuting the matrices. This could be a list of integers representing
            specific indices or a slice object to permute the matrices.

        Notes
        -----
        - This method operates on both the `TRAIN` and `TEST` datasets, which are instances of the `Dataset` class.
        - It calls the `permute_matrices` method on both the `TRAIN` and `TEST` datasets individually.
        """
        self.TRAIN.permute_matrices(permutation_idx=permutation_idx)
        self.TEST.permute_matrices(permutation_idx=permutation_idx)

    def train_test_split(self, test_size, seed):
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        test_size : float or int
            Proportion of the dataset to include in the test split if a float, or the absolute number of test samples if an integer.
        seed : int
            Random seed for reproducibility of the split.

        Returns
        -------
        None
            This method does not return any values. Instead, it assigns the split datasets to the `TRAIN` and `TEST` attributes of the instance.
        """
        if isinstance(test_size, int):
            # convert to train_test ratio
            assert test_size < self.n_sim
            test_size = test_size / self.n_sim

        X, X_test, X_dt, X_dt_test = train_test_split(
            self.X, self.X_dt, test_size=test_size, random_state=seed
        )

        # inputs
        if self.U is not None:
            U, U_test = train_test_split(self.U, test_size=test_size, random_state=seed)
        else:
            U, U_test = None, None
        # parameters
        if self.Mu is not None:
            Mu, Mu_test = train_test_split(
                self.Mu, test_size=test_size, random_state=seed
            )
        else:
            Mu, Mu_test = None, None
        # input parameters
        if self.Mu_input is not None:
            Mu_input, Mu_input_test = train_test_split(
                self.Mu_input, test_size=test_size, random_state=seed
            )
        else:
            Mu_input, Mu_input_test = None, None
        # ph matrices (reshape due to (r,r,n_sim) structure)
        if self.J is not None:
            J, J_test = train_test_split(self.J, test_size=test_size, random_state=seed)
        else:
            J, J_test = None, None
        if self.R is not None:
            R, R_test = train_test_split(self.R, test_size=test_size, random_state=seed)
        else:
            R, R_test = None, None
        if self.Q is not None:
            Q, Q_test = train_test_split(self.Q, test_size=test_size, random_state=seed)
        else:
            Q, Q_test = None, None
        if self.B is not None:
            B, B_test = train_test_split(self.B, test_size=test_size, random_state=seed)
        else:
            B, B_test = None, None

        self.TRAIN = Data(
            self.t,
            X,
            U=U,
            X_dt=X_dt,
            Mu=Mu,
            J=J,
            R=R,
            Q=Q,
            B=B,
            Mu_input=Mu_input,
        )

        self.TEST = Data(
            self.t,
            X_test,
            U=U_test,
            X_dt=X_dt_test,
            Mu=Mu_test,
            J=J_test,
            R=R_test,
            Q=Q_test,
            B=B_test,
            Mu_input=Mu_input_test,
        )

        # reset data from Dataset level to omit confusion
        self.t, self.X, self.U, self.X_dt, self.Mu, self.J, self.R, self.Q, self.B = [
            None
        ] * 9

    def train_test_split_sim_idx(self, sim_idx_train, sim_idx_test):
        """
        Manually split the data into training and testing sets based on simulation indices.

        Parameters
        ----------
        sim_idx_train : list or array-like
            List or array of indices for selecting the training simulations.
        sim_idx_test : list or array-like
            List or array of indices for selecting the testing simulations.

        Notes
        -----
        - The method ensures that there are no overlapping indices between the training and testing sets.
        - The method updates the `TRAIN` and `TEST` attributes with the corresponding subsets of data, including states, time derivatives, inputs, parameters, and pH matrices.
        """
        # no common element in data and test set
        assert not (set(sim_idx_test) & set(sim_idx_train))

        # states
        X_test = self.X[sim_idx_test]
        X = self.X[sim_idx_train]
        # states time derivative
        X_dt_test = self.X_dt[sim_idx_test]
        X_dt = self.X_dt[sim_idx_train]
        # input
        U_test = self.U[sim_idx_test]
        U = self.U[sim_idx_train]

        # parameters
        if self.Mu is not None:
            Mu_test = self.Mu[sim_idx_test]
            Mu = self.Mu[sim_idx_train]
        else:
            Mu_test, Mu = [None] * 2

        # input parameters
        if self.Mu_input is not None:
            Mu_input_test = self.Mu_input[sim_idx_test]
            Mu_input = self.Mu_input[sim_idx_train]
        else:
            Mu_input_test, Mu_input = [None] * 2

        # ph matrices
        if self.J is not None:
            J_test = self.J[sim_idx_test, :, :]
            J = self.J[sim_idx_train, :, :]
        else:
            J_test, J = [None] * 2
        if self.R is not None:
            R_test = self.R[sim_idx_test, :, :]
            R = self.R[sim_idx_train, :, :]
        else:
            R_test, R = [None] * 2
        if self.Q is not None:
            Q_test = self.Q[sim_idx_test, :, :]
            Q = self.Q[sim_idx_train, :, :]
        else:
            Q_test, Q = [None] * 2
        if self.B is not None:
            B_test = self.B[sim_idx_test, :, :]
            B = self.B[sim_idx_train, :, :]
        else:
            B_test, B = [None] * 2

        self.TRAIN = Data(
            self.t,
            X,
            U=U,
            X_dt=X_dt,
            Mu=Mu,
            J=J,
            R=R,
            Q=Q,
            B=B,
            Mu_input=Mu_input,
        )

        self.TEST = Data(
            self.t,
            X_test,
            U=U_test,
            X_dt=X_dt_test,
            Mu=Mu_test,
            J=J_test,
            R=R_test,
            Q=Q_test,
            B=B_test,
            Mu_input=Mu_input_test,
        )

        # save indices as attributes
        self.TRAIN.sim_idx = sim_idx_train
        self.TEST.sim_idx = sim_idx_test

        # reset data from Dataset level to omit confusion
        self.t, self.X, self.U, self.X_dt, self.Mu, self.J, self.R, self.Q, self.B = [
            None
        ] * 9

    def train_test_split_convex_hull(
        self,
        desired_min_num_train: int = 1,
        n_simulations_per_parameter_set: int = 1,
        plot_convex_hull: bool = False,
    ):
        """
        Split the dataset into training and testing sets based on the convex hull of the parameter space.

        This method divides the dataset into training and testing subsets by first identifying the convex hull
        of the parameter space (`Mu`). The training set is defined by the vertices of the convex hull, and
        the testing set consists of the remaining points. The number of training samples is adjusted to meet
        the `desired_min_num_train`, by shifting some of the testing set data into the training set if necessary.
        Optionally, the convex hull can be visualized in a 3D plot.

        Parameters
        ----------
        desired_min_num_train : int, optional
            The desired minimum number of training samples. If the training set has fewer samples than this value,
            samples from the testing set are moved to the training set. Default is 1.
        n_simulations_per_parameter_set : int, optional
            The number of simulations per parameter set. This is used to adjust the training and testing sets.
            Default is 1.
        plot_convex_hull : bool, optional
            If True, a 3D plot of the convex hull will be displayed. Default is False.

        Raises
        ------
        ValueError
            If there are not enough parameter points in the test set to shift to the training set, a ValueError
            is raised.
        NotImplementedError
            If `Mu` has more than 3 dimensions, a NotImplementedError is raised when trying to plot the convex hull.

        Notes
        -----
        - The training and testing sets are created by performing a convex hull operation on the `Mu` parameters.
        - The `train_test_split_sim_idx` method is called to split the simulations into training and testing datasets.
        - A 3D plot of the convex hull is generated if `plot_convex_hull` is set to `True` and `Mu` has 3 dimensions.
        """
        _, idx = np.unique(self.Mu, axis=0, return_index=True)
        Mu_all = self.Mu[np.sort(idx), :]
        hull_convex = ConvexHull(Mu_all)
        idx_train = hull_convex.vertices  # parameters that define the convex hull
        idx_test = np.setdiff1d(
            np.arange(Mu_all.shape[0]), idx_train
        )  # parameters inside the convex hull

        train_idx = np.array(
            [
                (i_same_mu_scenario) * n_simulations_per_parameter_set
                + np.arange(n_simulations_per_parameter_set)
                for i_same_mu_scenario in idx_train
            ]
        ).flatten()
        test_idx = np.array(
            [
                (i_same_mu_scenario) * n_simulations_per_parameter_set
                + np.arange(n_simulations_per_parameter_set)
                for i_same_mu_scenario in idx_test
            ]
        ).flatten()
        # shift values to training set based on desired_min_num_train
        if len(train_idx) < desired_min_num_train:
            shift_to_train = desired_min_num_train - len(train_idx)
            if shift_to_train >= len(test_idx):
                raise ValueError(
                    f"There are not enough parameter points in mu_test to shift to the training set."
                )
            if np.mod(shift_to_train, n_simulations_per_parameter_set) != 0:
                shift_to_train = (
                    np.ceil(shift_to_train / n_simulations_per_parameter_set)
                    * n_simulations_per_parameter_set
                )
                logging.info(
                    f"Desired number of training data can't be achieved, use {shift_to_train + len(train_idx)}"
                )
            train_idx = np.concatenate([train_idx, test_idx[:shift_to_train]])
            test_idx = test_idx[shift_to_train:]
        # train and test split
        self.train_test_split_sim_idx(sim_idx_train=train_idx, sim_idx_test=test_idx)

        if plot_convex_hull:
            if Mu_all.shape[1] != 3:
                raise NotImplementedError(
                    f"Convex hull plot currently only implemented for the 3D case, i.e. n_mu=3."
                )
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(Mu_all[:, 0], Mu_all[:, 1], Mu_all[:, 2], "b")
            ax.scatter(
                Mu_all[:, 0],
                Mu_all[:, 1],
                Mu_all[:, 2],
                "r",
            )
            for s in hull_convex.simplices:
                s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                ax.plot(
                    Mu_all[s, 0],
                    Mu_all[s, 1],
                    Mu_all[s, 2],
                    "r-",
                )
            plt.show()

    def truncate_time(self, trunc_time_ratio):
        """
        Truncates the time values of states for performing time generalization experiments.
        Only training data is truncated!

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
        logging.info(
            f"Only the time values of the training data set are reduced with truncation value {trunc_time_ratio}."
        )
        self.TRAIN.truncate_time(trunc_time_ratio)

    def cut_time_start_and_end(self, num_cut_idx=5):
        """
        Cut the start and end of the time series for both the training and testing datasets.

        This method removes a specified number of time steps from the beginning and end of the time series
        in both the training and testing datasets. The number of time steps to be removed is controlled by
        the `num_cut_idx` parameter.

        Parameters
        ----------
        num_cut_idx : int, optional
            The number of time steps to be removed from the start and end of the time series. Default is 5.

        Notes
        -----
        - This method applies the same cut to both the training and testing datasets.
        - The method is designed to adjust the time series for better alignment or to remove irrelevant portions
        of the data at the start and end of the time series.
        """
        self.TRAIN.cut_time_start_and_end(num_cut_idx=num_cut_idx)
        self.TEST.cut_time_start_and_end(num_cut_idx=num_cut_idx)

    def remove_mu(self):
        """
        Removes the 'mu' parameter from both the training and testing datasets.

        This method removes the 'mu' parameter from the training and testing datasets,
        effectively discarding any parameter-dependent data associated with 'mu'.

        Notes
        -----
        - This operation is applied to both the training and testing datasets.
        - Use this method when 'mu' is no longer needed for the subsequent processing or analysis.

        """
        self.TRAIN.remove_mu()
        self.TEST.remove_mu()

    def decrease_num_simulations(
        self,
        num_sim: int | None = None,
        seed: int | None = None,
        sim_idx: list[int] | np.ndarray | None = None,
    ):
        """
        Reduce the number of training simulations to a specified target number by randomly selecting a subset or defining specified indices.
        The method modifies the `TRAIN` dataset only, leaving the `TEST` dataset unaffected.

        Parameters
        ----------
        num_sim : int
            The target number of simulations to retain. If `None`, the number of simulations is not altered.
        seed : int, optional
            Random seed for reproducibility of the selection process. If not provided, a random seed is used.
        sim_idx : list of int, np.ndarray, or None, optional
            The indices of simulations to retain. If `None`, simulations are selected randomly.

        Notes
        -----
        - This method only affects the training dataset (`TRAIN`), not the test dataset (`TEST`).
        - The parameter `sim_idx` can be used to specify which simulations to retain, overriding the random selection.
        - The random seed ensures that the subset selection is reproducible.
        """
        # if self.X_test is not None:
        logging.info(
            f"Only the number of the training data set is reduced to {num_sim}."
        )
        self.TRAIN.decrease_num_simulations(num_sim=num_sim, seed=seed, sim_idx=sim_idx)

    def decrease_num_time_steps(self, num_time_steps: int):
        """
        Truncate the number of time steps in both the training and testing datasets to the specified target.

        Parameters
        ----------
        num_time_steps : int
            The target number of time steps to retain in the datasets.

        Notes
        -----
        - This method applies the truncation to both the training dataset (`TRAIN`) and the testing dataset (`TEST`).
        """
        self.TRAIN.decrease_num_time_steps(num_time_steps=num_time_steps)
        self.TEST.decrease_num_time_steps(num_time_steps=num_time_steps)

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

        Notes
        -----
        - The transformation applies to both the training and testing datasets.
        """
        self.TRAIN.states_to_features()
        self.TEST.states_to_features()

    def features_to_states(self):
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
        Notes
        -----
        - The transformation applies to both the training and testing datasets.
        """
        self.TRAIN.features_to_states()
        self.TEST.features_to_states()

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
        Scales states `X`, inputs `U`, and parameters `Mu`.

        This method applies scaling to the states, inputs, and parameters of both the training and testing datasets.
        It first scales the state data using the specified scaling values and domain splits, then scales the inputs
        and parameters according to the given bounds.

        Parameters:
        -----------
        scaling_values : list of float, optional
            Scalar values used to scale each domain of the state data. If `None`, scaling is based on the
            maximum value of each domain. See `scale_X` for more details.

        domain_split_vals : list of int, optional
            List specifying the number of degrees of freedom (DOFs) in each domain. The sum of these values
            must equal the total number of DOFs per node (`self.n_dn`). If `None`, the data is treated as a
            single domain. See `scale_X` for more details.

        u_train_bounds : np.ndarray, optional
            Scaling factors obtained from the training parameter dataset. Expected shape is (2, n_u).
            The lower and upper bounds of the training input data `U`. If `None`, the bounds are automatically
            determined from the training data.

        u_desired_bounds : list or scalar, optional
            Desired range for the scaled inputs. Can be:
            - A list of two values (e.g., [-1, 1]) to specify the lower and upper bounds.
            - A scalar to scale all input values by this value.
            - The string "max" to scale all input values by the maximum value observed. Default is `[-1, 1]`.

        mu_train_bounds : np.ndarray, optional
            Scaling factors obtained from the training parameter dataset. Expected shape is (2, n_mu).
            The lower and upper bounds of the training parameter data `Mu`. If `None`, the bounds are automatically
            determined from the training data.

        mu_desired_bounds : list, scalar, or "max", optional
            Desired range for the scaled values. It can be:
            - A list of two values (e.g., [-1, 1]) specifying the lower and upper bounds.
            - A scalar, where all values are scaled by this single value.
            - The string "max", where scaling is based on the maximum value in `Mu`. Default is `[-1, 1]`.

        Returns:
        --------
        None
            The method updates the internal state, input, and parameter data to reflect the scaled values.

        Notes:
        ------
        - This method scales the state data `X`, input data `U`, and parameter data `Mu` of both the training
        and testing datasets.
        - After scaling, the feature representation of the states is updated using `self.scale_X`.
        - The scaling for inputs and parameters is performed based on the specified or automatically determined
        bounds.
        """
        self.scale_X(scaling_values=scaling_values, domain_split_vals=domain_split_vals)
        self.scale_U(u_train_bounds=u_train_bounds, desired_bounds=u_desired_bounds)
        self.scale_Mu(mu_train_bounds=mu_train_bounds, desired_bounds=mu_desired_bounds)

    def scale_X(self, scaling_values=None, domain_split_vals=None):
        """
        Scale the state array based on specified scaling values.
        The scaling is applied to both the training and testing datasets.

        This method scales the state array `X` with shape (n_sim, n_t, n_n, n_dn) and afterwards the feature array `x`
        with shape (n_t * n_s, n) using provided scaling values. It can handle multiple domains if specified
        by `domain_split_vals`. If scaling values are not provided, it defaults to scaling by the maximum value
        in each domain.

        Parameters:
        -----------
        scaling_values : list of float, optional
            Scalar values used to scale each domain, as defined by `domain_split_vals`. If `None`, scaling
            values of the training dataset are used for the testing dataset

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
        self.TRAIN.scale_X(scaling_values, domain_split_vals)
        if scaling_values is None:
            scaling_values = self.TRAIN.scaling_values
        self.TEST.scale_X(scaling_values, domain_split_vals)

    def rescale_X(self):
        """
        Undo the scaling of states and time derivatives.
        The scaling is applied to both the training and testing datasets.

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
        self.TRAIN.rescale_X()
        self.TEST.rescale_X()

    def scale_Mu(self, mu_train_bounds=None, desired_bounds=[-1, 1]):
        """
        Scale parameter values to the specified range and return the scaled values along with scaling factors.
        The scaling is applied to both the training and testing datasets.

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
            the data and scaling values from the training dataset are used for the testing dataset.

        Returns:
        --------
        None
            The method updates the internal state to reflect the rescaled parameters.
        """
        if self.TRAIN.Mu is not None:
            self.TRAIN.scale_Mu(mu_train_bounds, desired_bounds)
            if mu_train_bounds is None:
                mu_train_bounds = self.TRAIN.mu_train_bounds
            if desired_bounds == "max":
                desired_bounds = self.TRAIN.desired_bounds_mu
            self.TEST.scale_Mu(mu_train_bounds, desired_bounds)

    def scale_U(
        self,
        u_train_bounds=None,
        desired_bounds=[-1, 1],
    ):
        """
        Scale input values to a specified range.
        The scaling is applied to both the training and testing datasets.

        This method scales the input values `U` to the range defined by `desired_bounds`, which is typically
        [-1, 1]. Each dimension of the input is scaled individually. The scaling factors are either provided
        through `u_train_bounds` or computed based on the provided data.

        Parameters:
        -----------
        u_train_bounds : np.ndarray, optional
            Scaling factors obtained from the training parameter dataset. Expected shape is (2, n_u), where
            n_u is the number of input dimensions. If provided, these factors are used for scaling the input values.
            If not provided, bounds are computed from the data and these values from the training dataset are used
            for the testing dataset.

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
        if self.TRAIN.U is not None:
            self.TRAIN.scale_U(u_train_bounds, desired_bounds)
            if u_train_bounds is None:
                u_train_bounds = self.TRAIN.u_train_bounds
            if desired_bounds == "max":
                desired_bounds = self.TRAIN.desired_bounds_u
            self.TEST.scale_U(u_train_bounds, desired_bounds)

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
        if self.TRAIN.U is not None:
            self.TRAIN.scale_U_domain_wise(
                input_domain_split_vals=input_domain_split_vals,
                input_scaling_values=input_scaling_values,
            )
            if input_scaling_values is None:
                input_scaling_values = self.TRAIN.input_scaling_values
            self.TEST.scale_U_domain_wise(
                input_domain_split_vals=input_domain_split_vals,
                input_scaling_values=input_scaling_values,
            )

    def reshape_inputs_to_features(self):
        """
        Reshapes the input arrays `U` from the training and testing datasets into feature arrays `u`.

        This method converts the input arrays `U` from both the training and testing datasets into feature arrays
        required for the identification process. The transformation reshapes the input arrays from a 3D shape
        (n_sim, n_t, n_u) into a 2D feature array (n_s, n_u), where n_s is the total number of samples
        (n_sim * n_t), and n_u is the number of input features.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            The method updates the `u` attribute of both the `TRAIN` and `TEST` datasets by reshaping
            their respective input arrays `U`.

        Notes:
        ------
        - This method relies on the `reshape_inputs_to_features` function to perform the reshaping.
        - The reshaping is only performed if the input array `U` is not `None` in both the training and testing datasets.
        """
        if self.TRAIN.U is not None:
            self.TRAIN.u = reshape_inputs_to_features(self.TRAIN.U)
            self.TEST.u = reshape_inputs_to_features(self.TEST.U)

    def split_state_into_domains(self, domain_split_vals):
        """
        Splits the state array into different domains based on the specified dimensions.
        The splitting is applied to both the training and testing datasets.

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
        X_dom_list: list of ndarray
            A list where each element is a state array of the training dataset corresponding to a specific domain. The length
            of the list equals the number of domains specified by `domain_split_vals`.
        X_dom_test_list: list of ndarray
            A list where each element is a state of the testing dataset array corresponding to a specific domain. The length
            of the list equals the number of domains specified by `domain_split_vals`.
        """
        X_dom_list = self.TRAIN.split_state_into_domains(domain_split_vals)
        X_dom_test_list = self.TEST.split_state_into_domains(domain_split_vals)
        return X_dom_list, X_dom_test_list

    def reproject_with_basis(
        self,
        V: np.ndarray | list[np.ndarray],
        idx: slice | list[slice] | None = None,
        pick_method: str = "all",
        pick_entry: None | list[int] | int | np.ndarray = None,
        seed: None | int = None,
    ):
        """
        Reprojects the training and testing datasets onto a given basis.

        This method projects both the training and testing datasets onto a specified basis `V`. The projection
        can be applied to selected indices, based on various selection methods, and can also include randomization
        via a seed for reproducibility.

        Parameters:
        -----------
        V : ndarray or list of ndarray
            The basis onto which the data is to be projected. This can be a single array or a list of arrays
            representing different basis for projection.

        idx : slice or list of slice, optional
            Specifies the indices or slices of the data to project. If `None`, the entire dataset is projected.

        pick_method : str, optional
            The method for selecting the data entries to project. Default is 'all', which selects all data.
            Other methods might involve more specific or random selections (e.g., "random").

        pick_entry : None, list of int, int, or ndarray, optional
            Specifies the particular entries to pick for projection. If `None`, the selection is based on `pick_method`.
            This can be a specific index, a list of indices, or an array of indices to select.

        seed : int, optional
            Random seed for reproducibility of the selection process when `pick_method` involves random selection.
            If `None`, no seed is set, and the randomization will vary.

        Returns:
        --------
        None
            The method updates the training and testing datasets by projecting them onto the given basis `V`.

        Notes:
        ------
        - This method performs the re-projection on both the training (`TRAIN`) and testing (`TEST`) datasets.
        - The projection is based on the provided basis `V`, and selection criteria are determined by `pick_method` and `pick_entry`.
        """
        self.TRAIN.reproject_with_basis(
            V,
            idx=idx,
            pick_method=pick_method,
            pick_entry=pick_entry,
            seed=seed,
        )
        self.TEST.reproject_with_basis(
            V,
            idx=idx,
            pick_method=pick_method,
            pick_entry=pick_entry,
            seed=seed,
        )

    def calculate_errors(
        self,
        ph_identified_data_instance,
        domain_split_vals=None,
        save_to_txt=False,
        result_dir=None,
    ):
        """
        Calculates and stores errors (RMS and latent) for both training and testing datasets.

        This method calculates the Root Mean Square (RMS) errors for both the states and latent variables
        between the true and predicted states, separately for the training and testing datasets.
        It also provides an option to save the computed error values to a text file.

        Parameters:
        -----------
        ph_identified_data_instance : Data
            An instance of the `Data` class containing the predicted state variables and latent variables
            (if applicable). This instance is used to compute errors against the true states stored in
            the current instance.

        domain_split_vals : list of int, optional
            List specifying the number of degrees of freedom (DOFs) for each domain. If provided, it splits
            the states into domains for more granular error analysis.

        save_to_txt : bool, optional
            If `True`, the computed error values will be saved to a text file. Default is `False`.

        result_dir : str, optional
            The directory where the error file will be saved. This is required if `save_to_txt` is `True`.

        Returns:
        --------
        None
            The method updates internal attributes to store the computed state errors and latent errors
            (if latent variables are present). If `save_to_txt` is `True`, the errors are saved to a text file.

        Notes:
        ------
        - The method computes the RMS errors for states and latent variables separately for the training and
        testing datasets (`TRAIN` and `TEST`).
        - If latent variables (`Z`) are present in the `ph_identified_data_instance`, latent errors are computed
        and stored.
        - The computed error values are optionally saved to a text file in the specified `result_dir`.
        """
        self.TRAIN.calculate_errors(
            ph_identified_data_instance.TRAIN, domain_split_vals=domain_split_vals
        )
        self.TEST.calculate_errors(
            ph_identified_data_instance.TEST, domain_split_vals=domain_split_vals
        )
        if save_to_txt:
            file_name = "mean_error_measures.txt"
            if result_dir is None:
                raise ValueError(
                    f"Input result_dir is required for save_to_txt={save_to_txt}."
                )
            with open(os.path.join(result_dir, file_name), "w") as text_file:
                print(
                    f"TRAIN state error mean: {self.TRAIN.state_error_mean}",
                    file=text_file,
                )
                print(
                    f"TRAIN latent error mean: {self.TRAIN.latent_error_mean}",
                    file=text_file,
                )
                print(
                    f"TEST state error mean: {self.TEST.state_error_mean}",
                    file=text_file,
                )
                print(
                    f"TEST latent error mean: {self.TEST.latent_error_mean}",
                    file=text_file,
                )

    def save_data_conc(self, data_dir, save_name):
        """
        Saves concatenated training and testing data to a compressed file.

        This method combines the training and testing datasets (`TRAIN` and `TEST`), including states (`X`),
        state derivatives (`X_dt`), inputs (`U`), parameters (`Mu`), and input parameters (`Mu_input`),
        and saves them as a compressed `.npz` file at the specified location.

        Parameters:
        -----------
        data_dir : str
            The directory where the data file will be saved.

        save_name : str
            The name to be used for the saved file (without file extension). The data will be saved with this name
            as a `.npz` file.

        Returns:
        --------
        None
            The method saves the concatenated data into a `.npz` compressed file at the specified location.

        Notes:
        ------
        - The training (`TRAIN`) and testing (`TEST`) data are concatenated along the first axis (simulation axis).
        - The resulting `.npz` file contains the time vector `t`, concatenated states `X`, state derivatives `X_dt`,
        inputs `U`, parameters `Mu`, and input parameters `Mu_input`.
        - The data is saved in a compressed format for efficient storage.
        """
        X = np.concatenate((self.TRAIN.X, self.TEST.X), axis=0)
        X_dt = np.concatenate((self.TRAIN.X_dt, self.TEST.X_dt), axis=0)
        U = np.concatenate((self.TRAIN.U, self.TEST.U), axis=0)
        Mu = np.concatenate((self.TRAIN.Mu, self.TEST.Mu), axis=0)
        Mu_input = np.concatenate((self.TRAIN.Mu_input, self.TEST.Mu_input), axis=0)

        t = self.TRAIN.t

        data_path = os.path.join(data_dir, f"{save_name}.npz")
        np.savez_compressed(data_path, t=t, X=X, U=U, Mu=Mu, Mu_input=Mu_input)

    def save_video_data(self, data_dir, data_name: str = "video_data"):
        """
        Saves the state data for training and testing datasets to create a video.

        This method saves the state data of both the training and testing datasets, optionally rescaling the data
        if it has been scaled. The saved data can then be used for creating a video representation of the states.

        Parameters:
        -----------
        data_dir : str
            Directory where the video data will be saved.

        data_name : str, optional
            The name of the saved data file. Default is "video_data". The file will be saved as `data_name.npz`.

        Returns:
        --------
        None
            The method saves the state data for both the training and testing datasets to a compressed `.npz` file
            in the specified directory.

        Notes:
        ------
        - If the data has been scaled (i.e., `is_scaled` is `True`), the data is rescaled before saving.
        - The saved `.npz` file contains the state data for both the training (`X_train`) and testing (`X_test`) datasets.
        - This method is useful for visualizing the state data in a video format.
        """
        # rescale data
        if self.TRAIN.is_scaled:
            self.TRAIN.rescale_X()
        if self.TEST.is_scaled:
            self.TEST.rescale_X()
        # save state data
        logging.info(f"Saving state data for video creation.")
        np.savez_compressed(
            os.path.join(data_dir, f"{data_name}.npz"),
            X_train=self.TRAIN.X,
            X_test=self.TEST.X,
        )


class PHIdentifiedDataset(Dataset):
    """
    Class representing the identified port-Hamiltonian data which was obtained through the aphin framework.
    Used to store the PHIdentifiedDataset(Data) class of the training and testing dataset.

    """

    def __init__(self):
        """
        Initializes a PHIdentifiedData instance with placeholder for TRAIN and TEST.
        """
        self.TRAIN = None
        self.TEST = None

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
        Create an instance of `PHIdentifiedDataset` from the identified pH system.
        Creates these instances under TRAIN and TEST.

        This method generates both the training and testing datasets by processing the results
        obtained from a port-Hamiltonian identification procedure. It uses the specified system
        layer, pH network, integrator type, and decomposition option to compute the relevant data.

        Parameters:
        -----------
        data : Dataset
            The dataset containing the raw data for training and testing.

        system_layer : PHLayer or PHQLayer
            The port-Hamiltonian system layer that defines the system matrices.

        ph_network : PHIN or APHIN
            The network responsible for the identification of the pH system.

        integrator_type : str, optional
            The type of integrator to use for time integration (default is "IMR").

        decomp_option : str, optional
            The decomposition option for the pH matrices (default is "lu").

        calc_u_midpoints : bool, optional
            Flag indicating whether to calculate midpoints of the input `U`. If `True`, it computes the midpoints for `U`,
            potentially useful for certain system identification tasks.

        **kwargs : dict, optional
            Additional parameters to pass to the identification process.

        Returns:
        --------
        PHIdentifiedDataset
            An instance of `PHIdentifiedDataset` containing the training and testing datasets
            with the identified pH system results.

        Notes:
        ------
        - This method logs the progress of obtaining results for both the training and testing datasets.
        - The `PHIdentifiedData.from_identification` method is called separately for training and testing datasets.
        """
        cls = PHIdentifiedDataset()
        logging.info("Obtain results from training data")
        cls.TRAIN = PHIdentifiedData.from_identification(
            data.TRAIN,
            system_layer,
            ph_network,
            integrator_type,
            decomp_option,
            calc_u_midpoints,
            **kwargs,
        )
        logging.info("Obtain results from test data")
        cls.TEST = PHIdentifiedData.from_identification(
            data.TEST,
            system_layer,
            ph_network,
            integrator_type,
            decomp_option,
            calc_u_midpoints,
            **kwargs,
        )
        return cls

    @classmethod
    def from_system_list(cls, system_list_train, system_list_test, data):
        """
        Creates an instance of `PHIdentifiedDataset` from a list of systems, processing both training and test datasets.

        This class method generates both the training and testing datasets by using lists of systems for the training and
        testing data. It utilizes the `PHIdentifiedData.from_system_list` method to process the provided system lists along
        with the corresponding data for training and testing.

        Parameters:
        -----------
        system_list_train : list
            A list of systems used for the training data.

        system_list_test : list
            A list of systems used for the testing data.

        data : Dataset
            The dataset containing the raw data for training and testing.

        Returns:
        --------
        PHIdentifiedDataset
            An instance of `PHIdentifiedDataset` containing the training and testing datasets processed with the provided system lists.

        Notes:
        ------
        - This method calls `PHIdentifiedData.from_system_list` for both the training and testing datasets separately.
        - The method assumes the provided `system_list_train` and `system_list_test` are appropriate for the corresponding datasets.
        """
        cls = PHIdentifiedDataset()
        cls.TRAIN = PHIdentifiedData.from_system_list(
            system_list=system_list_train,
            data=data.TRAIN,
        )
        cls.TEST = PHIdentifiedData.from_system_list(
            system_list=system_list_test,
            data=data.TEST,
        )
        return cls


class DiscBrakeDataset(Dataset):
    """
    A dataset class specifically for handling data related to the linear thermoelastic disc brake model.
    """

    def __init__(
        self,
        t,
        X,
        X_dt=None,
        U=None,
        Mu=None,
        use_velocities=False,
        use_savgol=False,
        **kwargs,
    ):
        """
        Initializes an instance of the DiscBrakeDataset class.

        Parameters:
        -----------
        t : ndarray
            Array of time steps with shape (n_t,).

        X : ndarray
            States array with shape (n_sim, n_t, n_n, n_dn).

        X_dt : ndarray, optional
            Time derivatives of the states with the same shape as X. If not provided,
            the time derivatives are automatically computed using numerical differentiation.
            Default is None.

        U : ndarray, optional
            Input array with shape (n_sim, n_t, n_u). Default is None.

        Mu : ndarray, optional
            Parameters array with shape (n_sim, n_mu). Default is None.

        use_velocities : bool, optional
            If True, the state array `X` will be augmented with velocity information
            by including the time derivatives of specific state variables. This will
            result in an increased number of degrees of freedom per node (`n_dn`).
            Default is False.

        use_savgol : bool, optional
            If `True`, the time derivatives of the states are computed using the Savitzky-Golay filter
            (a smoothing and differentiation technique). Default is `False`.

        **kwargs : dict, optional
            Additional arguments to pass to the parent `Dataset` class.
        """

        if X_dt is None:
            # Compute time derivatives
            if use_savgol:
                logging.info(f"Using Savitzki Golay filter to calculate X_dt.")
                X_dt = scipy.signal.savgol_filter(
                    X,
                    window_length=20,
                    polyorder=1,
                    deriv=1,
                    axis=1,
                    delta=t.ravel()[1] - t.ravel()[0],
                )
            else:
                X_dt = np.gradient(X, t.ravel(), axis=1)
        else:
            pass

        self.use_velocities = use_velocities
        if use_velocities:
            logging.info(f"Converting to states with velocities included.")
            velocity_idx = range(1, 4)  # velocities
            X = np.concatenate((X, X_dt[:, :, :, velocity_idx]), axis=3)
            if use_savgol:
                X_ddt = scipy.signal.savgol_filter(
                    X_dt[:, :, :, velocity_idx],
                    window_length=20,
                    polyorder=1,
                    deriv=1,
                    axis=1,
                    delta=t.ravel()[1] - t.ravel()[0],
                )
            else:
                X_ddt = np.gradient(X_dt[:, :, :, velocity_idx], t.ravel(), axis=1)
            X_dt = np.concatenate((X_dt, X_ddt), axis=3)
        super().__init__(t, X, X_dt, U, Mu, **kwargs)

    @classmethod
    def from_data(
        cls,
        data_path,
        use_velocities=False,
        use_savgol=False,
        num_time_steps: int | None = None,
        **kwargs,
    ):
        """
        Reads data from a .npz file and returns it as a dictionary.

        Parameters:
        -----------
        data_path : str
            Path to the .npz file or the directory containing the .npz file. If a directory
            is provided, the method searches for the first .npz file in the directory.

        use_velocities : bool, optional
            If `True`, the state array will be augmented with velocity information by including
            the time derivatives of specific state variables. Default is `False`.

        use_savgol : bool, optional
            If `True`, the time derivatives of the states are computed using the Savitzky-Golay filter
            (a smoothing and differentiation technique). Default is `False`.

        num_time_steps : int, optional
            Number of time steps to load from the data. If `None`, all available time steps are loaded.

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
        data_dict = cls.read_data_from_npz(data_path, num_time_steps=num_time_steps)

        return cls(
            **data_dict, use_velocities=use_velocities, use_savgol=use_savgol, **kwargs
        )

    @classmethod
    def from_txt(
        cls,
        txt_path,
        idx_mu=None,
        n_t=None,
        t_start=0.0,
        t_end=None,
        save_cache=False,
        cache_path=None,
        use_velocities=False,
        **kwargs,
    ):
        """
        Load a disc brake dataset from .txt files generated by Abaqus and postprocessed with Abaqus-Python.

        This method reads temperature and displacement values for all nodes from a specified directory containing
        .txt files obtained from the Abaqus field outputs. The method handles the parsing and processing of these files,
        including optional downsampling of time steps, extraction of parameters that influence the system, and inclusion
        of additional input data such as force values.

        Parameters:
        -----------
        txt_path : str
            Path to the folder containing the .txt files generated by Abaqus.

        idx_mu : array-like, optional
            Index numbers of the columns corresponding to parameters (not inputs) that influence the system.
            If `None`, parameter extraction is skipped. Default is `None`.

        n_t : int, optional
            Number of time steps after downsampling. If `None`, all time steps are used. Default is `None`.

        t_start : float, optional
            The starting time for the data. Data before this time is discarded. Default is `0.0`.

        t_end : float, optional
            The ending time for the data. Data after this time is discarded. If `None`, no limit is set. Default is `None`.

        save_cache : bool, optional
            If `True`, the processed dataset will be saved to a cache file at `cache_path`. Default is `False`.

        cache_path : str, optional
            Path where the cached dataset will be saved if `save_cache` is `True`. Default is `None`.

        use_velocities : bool, optional
            If `True`, the dataset will include velocity information by augmenting the state arrays. Default is `False`.

        **kwargs : dict, optional
            Additional arguments passed to the `DiscBrakeDataset` constructor.

        Returns:
        --------
        DiscBrakeDataset
            An instance of the `DiscBrakeDataset` class containing the loaded and processed data.

        Notes:
        ------
        - This method handles reading `.txt` files that represent temperature and displacement data from a disc brake model,
          typically generated by Abaqus and processed using Abaqus-Python.
        - The method checks for the presence of `force` input files and incorporates them if available.
        - The dataset includes time steps `t`, states `X`, input `U`, and parameter information `Mu` (if available).
        - The processed data may be saved to a cache file to speed up future access.
        """
        t, X, Mu = None, None, None

        logging.info(f"Reading disc brake data from .txt files at {txt_path}.")
        # get names of all txt files in folder 'txt_path'
        file_name_list = []
        for file in os.listdir(txt_path):
            if (
                file.endswith(".txt")
                and file.startswith("field_output_")
                and not file.startswith("field_output_force")
            ):
                file_name_list.append(file)
        if not file_name_list:
            raise ValueError(f"No .txt files could be found in {txt_path}.")
        file_name_list = natsorted(
            file_name_list
        )  # keep samples in the same order (natural sorting - like in explorer)

        # get force input list
        force_file_name_list = []
        for file in os.listdir(txt_path):
            if file.endswith(".txt") and file.startswith("field_output_force"):
                force_file_name_list.append(file)
        if not force_file_name_list:
            force_input_exists = False
            logging.warning(
                f"Could not find any txt files starting with 'field_output_force' in {txt_path}."
            )
        else:
            force_input_exists = True

        # parameter information
        with open(
            os.path.join(txt_path, "parameter_information.txt")
        ) as txt_param_file:
            df_param = pd.read_csv(txt_param_file, delimiter=" ", engine="python")
        if idx_mu is not None:
            Mu = []
            df_param_numpy = df_param.iloc[:, idx_mu].to_numpy()
            # convert to specified format - should be the same as in input file
            convert_to_format = False  # for new data generation, the right format is saved to parameter_information
            if convert_to_format:
                specify_format = "{:0.2e}"
                logging.info(
                    f"Old data generation. Parameter values are reformatted to {specify_format}."
                )
                with np.nditer(df_param_numpy, op_flags=["readwrite"]) as it:
                    for df_param_numpy_value in it:
                        df_param_numpy_value[...] = float(
                            specify_format.format(df_param_numpy_value)
                        )
            else:
                # keep format of df_param_numpy
                logging.info(
                    "New data generation. Parameter values are not reformatted."
                )

        else:
            Mu = None

        # number of simulations
        n_sim = len(file_name_list)
        heat_flux_list = []
        force_freq_list = []
        for i_sim, file in enumerate(tqdm(file_name_list)):
            with open(os.path.join(txt_path, file_name_list[i_sim])) as txt_file:
                #     # _ = txt_file.readline()  # empty line
                df = pd.read_csv(txt_file, header=None, delimiter=" ", engine="python")
                # first row with node numbers
                row_node_num = 0
                # second row with degrees of freedom
                row_node_dof = 1
                # first column with time values
                if i_sim == 0:
                    header_num = 2  # number of node and number of dof
                    t_idx_start = np.argmax(df.to_numpy()[:, 0] >= t_start) + header_num
                    if t_end is None:
                        t_idx_end = df.shape[0] - 1
                    else:
                        t_idx_end = np.argmax(df.to_numpy()[:, 0] >= t_end) + header_num
                    if n_t is None:
                        t_idx = np.arange(t_idx_start, t_idx_end + 1, dtype=int)
                        n_t = len(t_idx)
                    else:
                        t_idx = np.linspace(t_idx_start, t_idx_end, n_t, dtype=int)
                    t = df.to_numpy()[t_idx, 0][:, np.newaxis]
                    t = t - t[0]

                    # states
                    n_n = int(np.max(df.to_numpy()[row_node_num, :]))
                    n_dn = int(
                        np.max(df.to_numpy()[row_node_dof, :])
                    )  # 1 temperature, 3 displacements, no velocity from Abaqus Standard
                    X = np.zeros((n_sim, n_t, n_n, n_dn))

                for i_col_df in range(
                    1, df.shape[1]
                ):  # start with 1 due to time column
                    idx_node_num = int(df.to_numpy()[row_node_num, i_col_df]) - 1
                    idx_node_dof = int(df.to_numpy()[row_node_dof, i_col_df]) - 1
                    X[i_sim, :, idx_node_num, idx_node_dof] = df.to_numpy()[
                        t_idx_start : t_idx_end + 1, i_col_df
                    ]

                # %% Input
                # get sample number from txt string
                sample_number = re.findall("sample\s*(\d+)", file_name_list[i_sim])

                # input
                if force_input_exists:
                    force_file_txt = [
                        i
                        for i in force_file_name_list
                        if i.endswith(f"sample{sample_number[0]}.txt")
                    ]
                    with open(
                        os.path.join(txt_path, force_file_txt[0])
                    ) as force_txt_file:
                        df_force = pd.read_csv(
                            force_txt_file, header=None, delimiter=" ", engine="python"
                        )
                        force_array = df_force.to_numpy()
                        # remove time column
                        force_array = np.delete(force_array, 0, axis=1)
                        # delete node and DOF rows which come from state conversion to .txt
                        force_array = np.delete(force_array, [0, 1], axis=0)
                        # remove inputs to come from Abaqus field outputs that are zero, e.g. forces in non-excited coordinate directions
                        idx_no_force = np.argwhere(
                            np.all(force_array[..., :] == 0, axis=0)
                        )
                        force_array = np.delete(force_array, idx_no_force, axis=1)
                        n_u = force_array.shape[1] + 1  # + 1 for heat flux
                else:
                    n_u = 1

                if i_sim == 0:
                    # initialize U
                    U = np.zeros((n_sim, n_t, n_u))

                heat_flux = df_param["heat_flux"]

                convert_to_format_input = False  # for new data generation, the right format is saved to parameter_information
                if convert_to_format_input:
                    # heat was created in the format ".e" and introduced in this format to the input file.
                    # reconvert the heat, in order to use the same input values
                    specify_format_input = "{:.0e}"
                    logging.info(
                        f"Old data generation. Input values are reformatted to {specify_format}."
                    )
                    U[i_sim, :, 0][:, np.newaxis] = np.ones((n_t, 1)) * float(
                        specify_format_input.format(
                            heat_flux[int(sample_number[0]) - 1]
                        )
                    )
                else:
                    logging.info(
                        f"New data generation. Input values are not reformatted."
                    )
                    U[i_sim, :, 0][:, np.newaxis] = (
                        np.ones((n_t, 1)) * heat_flux[int(sample_number[0]) - 1]
                    )

                if force_input_exists:
                    U[i_sim, :, 1:] = force_array[
                        t_idx_start - header_num : t_idx_end - header_num + 1
                    ]

                # get parameters
                if idx_mu is not None:
                    # mu = np.append(mu,df_param.iloc[int(sample_number[0])-1,idx_mu].to_numpy()[:,np.newaxis].T,axis=0)
                    Mu.append(
                        df_param_numpy[int(sample_number[0]) - 1, idx_mu][
                            :, np.newaxis
                        ].T
                    )
                heat_flux_list.append(heat_flux[int(sample_number[0]) - 1])
                if force_input_exists:
                    force_freq = df_param["force_freq"]
                    force_freq_list.append(force_freq[int(sample_number[0]) - 1])

        if idx_mu is not None:
            Mu = np.squeeze(np.array(Mu))

        if force_input_exists:
            Mu_input = np.concatenate(
                (
                    np.array(heat_flux_list)[:, np.newaxis],
                    np.array(force_freq_list)[:, np.newaxis],
                ),
                axis=1,
            )
        else:
            Mu_input = np.array(heat_flux_list)[:, np.newaxis]

        if save_cache:
            cls.save_data(cache_path, t, X, U, Mu=Mu, Mu_input=Mu_input)
        return cls(t=t, X=X, U=U, Mu=Mu, use_velocities=use_velocities, **kwargs)


class SynRMDataset(Dataset):
    def __init__(
        self, t, X, X_dt=None, U=None, Mu=None, J=None, R=None, Q=None, B=None
    ):
        """
        Initialize the SynRMDataset object.

        Parameters:
        -----------
        t : np.ndarray
            Time values for the dataset.

        X : np.ndarray
            The state matrix containing the states at each time step.

        X_dt : np.ndarray, optional
            The time derivative of the states, by default None.

        U : np.ndarray, optional
            Input forces or excitations, by default None.

        Mu : np.ndarray, optional
            System parameters influencing the state, by default None.

        J : np.ndarray, optional
            Interconnection matrix, by default None.

        R : np.ndarray, optional
            Dissipation matrix, by default None.

        Q : np.ndarray, optional
            Energy matrix, by default None.

        B : np.ndarray, optional
            Matrix for external input/output, by default None.
        """
        super().__init__(t, X, X_dt, U, Mu, J, R, Q, B)

    @classmethod
    def from_matlab(
        cls,
        data_path,
        return_V: bool = False,
        return_B: bool = False,
        exclude_states: str | None = None,
        scale_modes_individually: bool = False,
    ):
        """
        Load the dataset from a .mat file.

        Parameters:
        -----------
        data_path : str
            Path to the .mat file containing the data.

        return_V : bool, optional
            If True, returns the V matrix from the .mat file, by default False.

        return_B : bool, optional
            If True, returns the B matrix from the corresponding 'B.mat' file, by default False.

        exclude_states : str or None, optional
            Specifies which states to exclude from the dataset. Options are:
            - "no_phi"
            - "no_rigid"
            - "no_velocities"
            - "only_elastic", by default None.

        scale_modes_individually : bool, optional
            If True, scales the modes (specified slice ranges) individually, by default False.

        Returns:
        --------
        SynRMDataset
            An instance of the SynRMDataset class containing the loaded and processed data.
        """
        if not os.path.isfile(data_path):
            raise ValueError(f"The given path does not lead to a file.")
        if not data_path.endswith(".mat"):
            raise ValueError(
                f"The given file is not a .mat file or the .mat extension is not given."
            )

        # load mat file
        mat = scipy.io.loadmat(data_path)
        if return_V:
            V = mat["V"]
            return V

        if return_B:
            only_path = os.path.dirname(data_path)
            path_to_B = os.path.join(only_path, "B.mat")
            mat_B = scipy.io.loadmat(path_to_B)
            B = mat_B["B"]
            return B

        U = mat["U"]
        X = mat["X"]
        X_dt = mat["DX_dt"]
        t = mat["time"]

        if scale_modes_individually:
            slice_modes = np.r_[slice(80, 100), slice(105, 125)]
            X[:, :, slice_modes] = X[:, :, slice_modes] / np.max(
                np.abs(X[:, :, slice_modes]), axis=(0, 1)
            )
            X_dt[:, :, slice_modes] = X_dt[:, :, slice_modes] / np.max(
                np.abs(X_dt[:, :, slice_modes]), axis=(0, 1)
            )

        if exclude_states == "no_phi":
            # remove phi from X and X_dt
            X = np.delete(X, slice(3, 75), axis=2)
            X_dt = np.delete(X_dt, slice(3, 75), axis=2)
        elif exclude_states == "no_rigid":
            # no rigid and Drigid
            X = np.delete(X, np.r_[slice(75, 80), slice(100, 105)], axis=2)
            X_dt = np.delete(X_dt, np.r_[slice(75, 80), slice(100, 105)], axis=2)
        elif exclude_states == "no_velocities":
            # no Drigid and Delastic
            X = np.delete(X, slice(100, 125), axis=2)
            X_dt = np.delete(X_dt, slice(100, 125), axis=2)
        elif exclude_states == "only_elastic":
            # elastic and Delastic
            X = np.delete(X, np.r_[slice(0, 80), slice(100, 105)], axis=2)
            X_dt = np.delete(X_dt, np.r_[slice(0, 80), slice(100, 105)], axis=2)
            # scale each mode individually

        # add dimension for node DOFs
        if X.ndim == 3:
            X = X[:, :, np.newaxis, :]  # move DOF all to n_dn for scaling
        if X_dt.ndim == 3:
            X_dt = X_dt[:, :, np.newaxis, :]  # move DOF all to n_dn for scaling

        if t.ndim == 2:
            t = np.squeeze(t)

        return cls(t=t, X=X, U=U, X_dt=X_dt)
