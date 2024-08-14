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

# own package
from aphin.utils.data import Data, PHIdentifiedData


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

    def __init__(self, t, X, X_dt, U=None, Mu=None, J=None, R=None, Q=None, B=None):
        super().__init__(t, X, X_dt, U, Mu, J, R, Q, B)
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

        # ph matrices
        if self.J is not None:
            J_test = self.J[sim_idx_test, :, :]
            J = self.J[sim_idx_train, :, :]
        if self.R is not None:
            R_test = self.R[sim_idx_test, :, :]
            R = self.R[sim_idx_train, :, :]
        if self.Q is not None:
            Q_test = self.Q[sim_idx_test, :, :]
            Q = self.Q[sim_idx_train, :, :]
        if self.B is not None:
            B_test = self.B[sim_idx_test, :, :]
            B = self.B[sim_idx_train, :, :]

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
        )

        # reset data from Dataset level to omit confusion
        self.t, self.X, self.U, self.X_dt, self.Mu, self.J, self.R, self.Q, self.B = [
            None
        ] * 9

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

    def decrease_num_simulations(self, num_sim: int, seed=None):
        """
        Reduce the number of training simulations to a specified target number by randomly selecting a subset.

        Parameters
        ----------
        num_sim : int
            The target number of simulations to retain.
        seed : int, optional
            Random seed for reproducibility of the selection process. If not provided, a random seed is used.

        Notes
        -----
        - This method only affects the training dataset (`TRAIN`), not the test dataset (`TEST`).
        """
        # if self.X_test is not None:
        logging.info(
            f"Only the number of the training data set is reduced to {num_sim}."
        )
        self.TRAIN.decrease_num_simulations(num_sim, seed=seed)

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

    def scale_U(self, u_train_bounds=None, desired_bounds=[-1, 1]):
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

    def calculate_errors(
        self,
        ph_identified_data_instance,
        domain_split_vals=None,
        save_to_txt=False,
        result_dir=None,
    ):
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
            **kwargs,
        )
        logging.info("Obtain results from test data")
        cls.TEST = PHIdentifiedData.from_identification(
            data.TEST,
            system_layer,
            ph_network,
            integrator_type,
            decomp_option,
            **kwargs,
        )
        return cls


class DiscBrakeDataset(Dataset):
    """
    A dataset class specifically for handling data related to the linear thermoelastic disc brake model.
    """

    def __init__(
        self, t, X, X_dt=None, U=None, Mu=None, use_velocities=False, **kwargs
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

        **kwargs : dict, optional
            Additional arguments to pass to the parent `Dataset` class.
        """

        if X_dt is None:
            # Compute time derivatives
            X_dt = np.gradient(X, t.ravel(), axis=1)
        else:
            pass

        self.use_velocities = use_velocities
        if use_velocities:
            logging.info(f"Converting to states with velocities included.")
            velocity_idx = range(1, 4)  # velocities
            X = np.concatenate((X, X_dt[:, :, :, velocity_idx]), axis=3)
            X_dt = np.gradient(X, t.ravel(), axis=1)

        super().__init__(t, X, X_dt, U, Mu, **kwargs)

    @classmethod
    def from_data(cls, data_path, use_velocities=False, **kwargs):
        """
        Reads data from a .npz file and returns it as a dictionary.

        Parameters:
        -----------
        data_path : str
            Path to the .npz file or the directory containing the .npz file. If a directory
            is provided, the method searches for the first .npz file in the directory.

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
        data_dict = cls.read_data_from_npz(data_path)

        return cls(**data_dict, use_velocities=use_velocities, **kwargs)

    @classmethod
    def from_txt(
        cls,
        txt_path,
        idx_mu=None,
        n_t=None,
        t_start=0.0,
        save_cache=False,
        cache_path=None,
        use_velocities=False,
        **kwargs,
    ):
        """
        Load a disc brake dataset from .txt files generated by Abaqus and postprocessed with Abaqus-Python.

        This method reads temperature and displacement values for all nodes from a specified directory containing
        .txt files obtained from the Abaqus field outputs. The method handles the parsing and processing of these files,
        including optional downsampling of time steps and extraction of parameters that influence the system. The order
        of the trajectories and parameters `Mu` may not match the sample numbers from the data files obtained after the
        postprocessing with Abaqus-Python.

        Parameters:
        -----------
        txt_path : str
            Path to the folder containing the .txt files.

        idx_mu : array-like, optional
            Index numbers of the columns corresponding to parameters (not inputs) that influence the system.
            If `None`, parameter extraction is skipped. Default is `None`.

        n_t : int, optional
            Number of time steps after downsampling. If `None`, all time steps are used. Default is `None`.

        t_start : float, optional
            The starting time for the data. Data before this time is discarded. Default is `0.0`.

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
        """
        t, X, Mu = None, None, None

        logging.info(f"Reading disc brake data from .txt files at {txt_path}.")
        # get names of all txt files in folder 'txt_path'
        file_name_list = []
        for file in os.listdir(txt_path):
            if file.endswith(".txt") and file.startswith("field_output_"):
                file_name_list.append(file)
        if not file_name_list:
            raise ValueError(f"No .txt files could be found in {txt_path}.")
        file_name_list = natsorted(
            file_name_list
        )  # keep samples in the same order (natural sorting - like in explorer)

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
                    t_idx_end = df.shape[0] - 1
                    if n_t is None:
                        t_idx = np.arange(t_idx_start, t_idx_end + 1, dtype=int)
                        n_t = len(t_idx)
                    else:
                        t_idx = np.linspace(t_idx_start, t_idx_end, n_t, dtype=int)
                    t = df.to_numpy()[t_idx, 0][:, np.newaxis]
                    t = t - t[0]

                    # input
                    n_u = 1
                    U = np.zeros((n_sim, n_t, n_u))

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
                        header_num:, i_col_df
                    ]
                # get sample number from txt string
                sample_number = re.findall("sample\s*(\d+)", file_name_list[i_sim])
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

                # get parameters
                if idx_mu is not None:
                    # mu = np.append(mu,df_param.iloc[int(sample_number[0])-1,idx_mu].to_numpy()[:,np.newaxis].T,axis=0)
                    Mu.append(
                        df_param_numpy[int(sample_number[0]) - 1, idx_mu][
                            :, np.newaxis
                        ].T
                    )

        if idx_mu is not None:
            Mu = np.squeeze(np.array(Mu))

        if save_cache:
            cls.save_data(cache_path, t, X, U, Mu=Mu)
        return cls(t=t, X=X, U=U, Mu=Mu, use_velocities=use_velocities, **kwargs)
