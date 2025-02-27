import os
import logging
import numpy as np
from scipy import linalg

from aphin.systems.ph_systems import PHSystem
from aphin.systems.lti_systems import LTISystem
from aphin.utils.transformations import (
    reshape_features_to_states,
    reshape_states_to_features,
)

from matplotlib import pyplot as plt
import matplotlib as mpl


class MSD:
    def __init__(
        self,
        system_list,
        system_type,
        Mu,
        parameter_information,
        n_mass,
        M_inv=None,
        M=None,
        input_vals=None,
    ):
        """
        Initializes the MSD (mass-spring-damper) class with system parameters and configurations.

        Parameters:
        -----------
        system_list : list
            List of systems (PHSystem or LTISystem) to be used in the simulation.
        system_type : str
            Type of system; options include "ph" for port-Hamiltonian, "ss" for state-space, or "2nd" for second-order.
        Mu : np.ndarray
            Matrix of system parameters including mass, stiffness, and damping.
        parameter_information : dict
            Additional parameter information related to the systems.
        n_mass : int
            Number of masses in the system.
        M_inv : np.ndarray, optional
            Inverse of the mass matrix (used for state-space systems). Default is None.
        M : np.ndarray, optional
            Mass matrix (used for state-space systems). Default is None.
        input_vals : array-like, optional
            Input values for the systems. Default is None.

        Attributes:
        -----------
        n_sim : int
            Number of simulations or systems.
        Q_is_identity : bool
            Indicates whether the Q matrix has been transformed to the identity matrix.
        """
        self.n_sim = len(system_list)
        self.n_mass = n_mass
        self.system_list = system_list
        self.system_type = system_type
        self.Mu = Mu
        self.parameter_information = parameter_information
        self.M_inv = M_inv
        self.M = M
        self.input_vals = input_vals

        self.Q_is_identity = False

    def transform_pH_to_Q_identity(self, solver="Q", seed=1):
        system_list_temp = self.system_list.copy()
        self.system_list = []
        self.Q_is_identity = True
        for i_system, system in enumerate(system_list_temp):
            if not (isinstance(system, PHSystem)):
                logging.info(
                    f"The system is not a PHSystem. It is of type {type(system)} and can therefore not be transformed to Q identity."
                )
                self.system_list.append(system)  # keep system
                self.Q_is_identity = False
            else:
                system_transformed, T, T_inv = system.transform_pH_to_Q_identity(
                    solver=solver,
                    seed=seed,
                )
                if i_system == 0:
                    # initialize transformation matrix
                    T_all = np.repeat(
                        np.eye(T.shape[0])[:, :, np.newaxis],
                        len(system_list_temp),
                        axis=2,
                    )
                    T_inv_all = np.repeat(
                        np.eye(T_inv.shape[0])[:, :, np.newaxis],
                        len(system_list_temp),
                        axis=2,
                    )
                T_all[:, :, i_system] = T
                T_inv_all[:, :, i_system] = T_inv

                if system_transformed.Q_is_identity is False:
                    logging.info(
                        f"Not all pH systems could be transformed to identity form."
                    )
                    self.Q_is_identity = False
                else:
                    # check that Q is identity
                    assert np.allclose(
                        system_transformed.Q_ph,
                        np.eye(system_transformed.Q_ph.shape[0]),
                    )
                self.system_list.append(system_transformed)
                self.T = T_all
                self.T_inv = T_inv_all

    def convert_initial_condition(self, parameter_input_instance):
        """
        To compare the differenty system description, we convert the same random initial conditions to match the appropriate coordinate system,
        i.e. [disp vel] for state-space, T-transformed for Q=I coordinates and no change for the pH variant (defined that the random variables are in the format [disp mom])

        Parameters:
        -----------
        parameter_input_instance : ParameterInput
            Instance containing initial conditions and other parameters.

        Returns:
        --------
        ParameterInput
            The updated parameter input instance with converted initial conditions.

        Notes:
        ------
        - Converts initial conditions based on whether Q is identity, or velocity coordinates are used.
        """
        z_init = parameter_input_instance.x0
        for i_system, system in enumerate(self.system_list):
            if self.Q_is_identity:
                # convert to Q-identity coordinates
                parameter_input_instance.x0[:, i_system] = (
                    self.T[:, :, i_system] @ z_init[:, i_system]
                )
            elif self.M_inv is not None:
                # convert to velocity coordinates
                parameter_input_instance.x0[:, i_system] = (
                    linalg.block_diag(
                        np.eye(int(system.n / 2)), self.M_inv[:, :, i_system]
                    )
                    @ z_init[:, i_system]
                )
            else:
                # no change to parameter_input_instance.x0
                pass

        return parameter_input_instance

    def solve(
        self, t, z_init, u=None, integrator_type="IMR", decomp_option="lu", debug=False
    ):
        """
        see LTISystem solve
        """
        # initialize solution array
        system = self.system_list[0]
        X = np.zeros((self.n_sim, t.shape[0], int(system.n / 2), 2))

        for i, system in enumerate(self.system_list):
            if u is not None:
                u_i = u[i]
                u_i = self.convert_input(u_i, t)
                if i == 0:
                    # initialize array of overall u
                    U = np.zeros((self.n_sim, t.shape[0], u_i.shape[2]))
                # set last one to zero due to midpoints
                U[i, :-1, :] = u_i
            else:
                u_i = None
                U = None
            # initial condition
            z_init_i = z_init[:, i]
            # time integration
            x = system.solve(
                t,
                z_init=z_init_i,
                u=u_i,
                integrator_type=integrator_type,
                decomp_option=decomp_option,
            )

            # use reshape function (attention be aware, that even though it its of size (n_sim,n_t,n_mass,n_dn)
            # the last entries are not displacement and velocities of each mass accordingly - "Fortran" reshaping would be needed)
            n_t = len(t)
            X[i, :, :, :] = reshape_features_to_states(
                x, n_sim=1, n_t=n_t, n_n=int(system.n / 2), n_dn=2
            )
            # X[i, :, :, 0] = x[0, :, : int(system.n / 2)]
            # X[i, :, :, 1] = x[0, :, int(system.n / 2) :]

        self.X = X
        self.t = t
        self.U = U

        if debug:
            self.plot_states(t, X)

    def plot_states(self, t, X):
        """
        Plots the states of the system over time for debugging purposes.

        Parameters:
        -----------
        t : np.ndarray
            Array of time steps.
        X : np.ndarray
            Solution states to be plotted.

        Returns:
        --------
        None

        Notes:
        ------
        - Saves the plot to a file in the "plots_debug" directory.
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        # default debug save location
        work_dir = os.path.dirname(__file__)
        debug_dir = os.path.join(work_dir, "plots_debug")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        n_t = t.shape[0]
        n_mass = X.shape[2]
        n_sim = X.shape[0]
        if n_sim > 6:
            num_lines = 6
        else:
            num_lines = n_sim

        fig, ax = plt.subplots(n_mass)
        fig.suptitle("state")
        for i_mass in range(n_mass):
            ax[i_mass].plot(
                t,
                np.reshape(
                    np.transpose(X[:num_lines, :, i_mass, :], (1, 0, 2)),
                    (n_t, num_lines * 2),
                ),
            )
            ax[i_mass].set_ylabel(f"m_{i_mass}")
        plt.xlabel("time [s]")
        plt.show(block=False)
        plt.savefig(os.path.join(debug_dir, "state_plot.png"))

    def convert_input(self, u, t):
        """
        convert input into the required PHSystem.solve format (n_sim,n_t,n_u)
        """
        n_t = t.shape[0]
        t_midpoints = 0.5 * (t[1:] + t[:-1])
        u = u(t_midpoints)[np.newaxis, :]
        # for MIMO systems
        if u.ndim > 1:
            u = u.T
        # add n_sim = 1 axis
        u = u[np.newaxis, :]

        assert u.shape[0] == 1
        assert u.shape[1] == n_t - 1

        return u

    def save(
        self, save_path=None, save_name: str | None = None, save_name_suffix: str = ""
    ):
        """
        Saves the current state of the MSD instance to a file.

        Parameters:
        -----------
        save_path : str, optional
            Directory path where the file will be saved. Default is None.
        save_name : str, optional
            Name of the file to save. Default is None.

        Returns:
        --------
        None

        Notes:
        ------
        - If save_path or save_name is not provided, defaults are used.
        - Saves different system matrices based on the system type.
        """
        if save_path is None:
            work_dir = os.path.dirname(__file__)
            save_path = os.path.join(work_dir, "..", "data")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        if save_name is None:
            # add Qeye
            if self.Q_is_identity:
                add_Q_identity = f"Qeye"
            else:
                add_Q_identity = f""
            # add input type
            if self.input_vals is None:
                add_input = "autonomous"
            elif isinstance(self.input_vals, int):
                add_input = "siso"
            elif isinstance(self.input_vals, list):
                add_input = "mimo"
            else:
                # unknown
                add_input = ""
            if not save_name_suffix == "":
                save_name_suffix = f"_{save_name_suffix}"

            save_name = f"MSD_{add_Q_identity}_{self.system_type}_input_{add_input}{save_name_suffix}"
            save_name = save_name.replace("__", "_")

        if self.system_type == "ph":
            J, R, B, Q = self.get_system_matrices()
            np.savez(
                os.path.join(save_path, save_name),
                X=self.X,
                t=self.t,
                U=self.U,
                J=J,
                R=R,
                B=B,
                Q=Q,
                Mu=self.Mu,  # only save mass, stiffness, damping
                parameter_information=self.parameter_information,
            )
        elif self.system_type == "ss":
            A, B, C = self.get_system_matrices()
            np.savez(
                os.path.join(save_path, save_name),
                X=self.X,
                t=self.t,
                U=self.U,
                A=A,
                B=B,
                C=C,
                Mu=self.Mu,  # only save mass, stiffness, damping
                parameter_information=self.parameter_information,
            )

    def get_system_matrices(self):
        """
        Retrieves the system matrices for all systems in the system list.

        Returns:
        --------
        tuple
            A tuple containing system matrices: (J, R, B, Q) for "ph" systems or (A, B, C) for "ss" systems.
        """
        for i_system, system in enumerate(self.system_list):
            if self.system_type == "ph":
                J, R, B, Q = system.get_system_matrix()
                if i_system == 0:
                    # initialize array of matrices
                    J_all = np.repeat(
                        np.zeros_like(J)[np.newaxis, :, :], self.n_sim, axis=0
                    )
                    R_all = np.repeat(
                        np.zeros_like(R)[np.newaxis, :, :], self.n_sim, axis=0
                    )
                    Q_all = np.repeat(
                        np.zeros_like(Q)[np.newaxis, :, :], self.n_sim, axis=0
                    )
                    B_all = np.repeat(
                        np.zeros_like(B)[np.newaxis, :, :], self.n_sim, axis=0
                    )
                J_all[i_system, :, :] = J
                R_all[i_system, :, :] = R
                B_all[i_system, :, :] = B
                Q_all[i_system, :, :] = Q
            elif self.system_type == "ss":
                A, B, C = system.get_system_matrix()
                if i_system == 0:
                    # initialize array of matrices
                    A_all = np.repeat(
                        np.zeros_like(A)[np.newaxis, :, :], self.n_sim, axis=0
                    )
                    B_all = np.repeat(
                        np.zeros_like(B)[np.newaxis, :, :], self.n_sim, axis=0
                    )
                    C_all = np.repeat(
                        np.zeros_like(C)[np.newaxis, :, :], self.n_sim, axis=0
                    )
                A_all[i_system, :, :] = A
                B_all[i_system, :, :] = B
                C_all[i_system, :, :] = C

        if self.system_type == "ph":
            return J_all, R_all, B_all, Q_all
        elif self.system_type == "ss":
            return A_all, B_all, C_all

    @classmethod
    def from_parameter_input(cls, parameter_input_instance, system_type):
        """
        Creates an MSD instance from a parameter input instance.

        Parameters:
        -----------
        parameter_input_instance : ParameterInput
            Instance containing system parameters and initial conditions.
        system_type : str
            Type of system to create ("ph", "ss", or "2nd").

        Returns:
        --------
        MSD
            A new instance of the MSD class initialized with the given parameters.
        """
        assert system_type in ["ph", "ss", "2nd"]

        n_sim = parameter_input_instance.n_sim
        n_mass = parameter_input_instance.n_mass

        # safe parameters to instance
        if (
            parameter_input_instance.mass_vals[:, 0].max()
            == parameter_input_instance.mass_vals[:, 0].min()
            and parameter_input_instance.stiff_vals[:, 0].max()
            == parameter_input_instance.stiff_vals[:, 0].min()
            and parameter_input_instance.damp_vals[:, 0].max()
            == parameter_input_instance.damp_vals[:, 0].min()
        ):
            # same parameter value for all masses
            Mu = np.concatenate(
                (
                    parameter_input_instance.mass_vals[0, :][:, np.newaxis],
                    parameter_input_instance.stiff_vals[0, :][:, np.newaxis],
                    parameter_input_instance.damp_vals[0, :][:, np.newaxis],
                ),
                axis=1,
            )
        else:
            Mu = None

        # initialize list with systems
        system_list = []

        # M inverse for state-space transformation
        M_inv_all = None
        M_all = None
        for i_sim in range(n_sim):
            mass_val = parameter_input_instance.mass_vals[:, i_sim]
            stiff_val = parameter_input_instance.stiff_vals[:, i_sim]
            damp_val = parameter_input_instance.damp_vals[:, i_sim]
            input_vals = parameter_input_instance.input_vals

            if system_type == "ph":
                J, R, B, Q = cls.create_mass_spring_damper_system(
                    n_mass,
                    mass_vals=mass_val,
                    damp_vals=damp_val,
                    stiff_vals=stiff_val,
                    input_vals=input_vals,
                    system=system_type,
                )

                system_list.append(PHSystem(J, R, B, Q))
                # C = B.T @ Q
                # A_ph = (J - R) @ Q
                # print(J, R, Q, B)
            elif system_type == "2nd":
                # for comparison
                M, D, K, B = cls.create_mass_spring_damper_system(
                    n_mass,
                    mass_vals=mass_val,
                    damp_vals=damp_val,
                    stiff_vals=stiff_val,
                    input_vals=input_vals,
                    system=system_type,
                )
                print(M, D, K, B)
            elif system_type == "ss":
                A, B, M = cls.create_mass_spring_damper_system(
                    n_mass,
                    mass_vals=mass_val,
                    damp_vals=damp_val,
                    stiff_vals=stiff_val,
                    input_vals=input_vals,
                    system=system_type,
                )
                C = B.T
                M_inv = np.linalg.solve(M, np.eye(M.shape[0]))
                if i_sim == 0:
                    M_inv_all = np.repeat(
                        np.zeros_like(M_inv)[:, :, np.newaxis], n_sim, axis=2
                    )
                    M_all = np.repeat(np.zeros_like(M)[:, :, np.newaxis], n_sim, axis=2)
                M_inv_all[:, :, i_sim] = M_inv
                M_all[:, :, i_sim] = M
                # print(A, B)
                system_list.append(LTISystem(A, B))

            # # get initial conditions
            # x0 = parameter_input_instance.x0
            # # get input
            # u = parameter_input_instance.u

        return cls(
            system_list,
            system_type,
            Mu,
            parameter_input_instance.parameter_information,
            n_mass,
            M_inv=M_inv_all,
            M=M_all,
            input_vals=input_vals,
        )

    @staticmethod
    def create_mass_spring_damper_system(
        n_mass,
        mass_vals=1,
        damp_vals=1,
        stiff_vals=1,
        input_vals=None,
        system="ph",
    ):
        """
        Generate the matrices for a mass-spring-damper system based on the specified system type.

        This function creates the system matrices for different representations of a mass-spring-damper system:
        - Port-Hamiltonian (ph)
        - Second-order (2nd)
        - State-space (ss)

        Parameters:
        - n_mass (int): Number of masses (also determines the size of the second-order system).
        - mass_vals (scalar or array-like, default=1): Mass values. Can be a scalar (applied to all masses) or an array of shape (n_mass,).
        - damp_vals (scalar or array-like, default=1): Damping values. Can be a scalar (applied to all dampers) or an array of shape (n_mass,).
        - stiff_vals (scalar or array-like, default=1): Stiffness values. Can be a scalar (applied to all springs) or an array of shape (n_mass,).
        - input_vals (array-like or int, optional): Defines the input configuration. If an int, a single input is created. If an array, it specifies the indices of excited masses.
        - system (str, default="ph"): Specifies the type of system matrices to return. Options are:
        - "ph": Port-Hamiltonian matrices (J, R, B_ph, Q)
        - "2nd": Second-order system matrices (M, D, K, B_2nd)
        - "ss": State-space matrices (A, B, M)

        Returns:
        - Depending on the `system` parameter:
        - For "ph": (J, R, B_ph, Q)
        - For "2nd": (M, D, K, B_2nd)
        - For "ss": (A, B, M)
        """

        # %% create second order system
        # create mass matrix
        if isinstance(mass_vals, (list, tuple, np.ndarray)):
            M = np.diag(mass_vals)
        else:  # scalar value
            M = np.eye(n_mass) * mass_vals

        # create damping matrix
        if isinstance(mass_vals, (list, tuple, np.ndarray)):
            D = np.diag(damp_vals)
        else:  # scalar value
            D = np.eye(n_mass) * damp_vals

        # create stiffness matrix
        if not isinstance(stiff_vals, (list, tuple, np.ndarray)):
            # scalar value
            stiff_vals = np.ones(n_mass) * stiff_vals
        K = np.zeros((n_mass, n_mass))
        K[:, :] = np.diag(stiff_vals[:])
        K[1:, 1:] += np.diag(stiff_vals[:-1])
        K += -np.diag(stiff_vals[:-1], -1)
        K += -np.diag(stiff_vals[:-1], 1)

        # create input vector
        if input_vals is not None:
            if isinstance(input_vals, int):
                # single input
                assert input_vals <= n_mass
                B_2nd = np.zeros((n_mass, 1))
                B_2nd[input_vals, 0] = 1
            else:
                assert max(list(input_vals)) <= n_mass
                B_2nd = np.zeros((n_mass, len(input_vals)))
                B_2nd[input_vals, np.arange(len(input_vals))] = 1
        else:
            B_2nd = np.zeros((n_mass, 1))

        # %% create state-space system
        M_inv = np.linalg.solve(M, np.eye(M.shape[0]))
        A = np.block(
            [[np.zeros((n_mass, n_mass)), np.eye(n_mass)], [-M_inv @ K, -M_inv @ D]]
        )

        # scale input with M
        B = np.concatenate((np.zeros(B_2nd.shape), M_inv @ B_2nd), axis=0)

        # %% convert to port-Hamiltonian system
        J = np.diag(np.ones(n_mass), n_mass)
        J += -np.diag(np.ones(n_mass), -n_mass)

        R = linalg.block_diag(np.zeros((n_mass, n_mass)), D)

        Q = linalg.block_diag(K, np.linalg.inv(M))

        # B_ph cancels out M with M_inv due to momentum description
        B_ph = np.concatenate((np.zeros(B_2nd.shape), B_2nd), axis=0)

        if system == "ph":
            return J, R, B_ph, Q
        elif system == "2nd":
            return M, D, K, B_2nd
        elif system == "ss":
            return A, B, M
        else:
            raise Exception("system input not known. Choose either 'ph','2nd' or 'ss' ")

    @classmethod
    def compare_msd_systems(cls, msd_list):
        """
        Compares multiple MSD systems and plots their states.

        Parameters:
        -----------
        msd_list : list
            List of MSD instances to compare.

        Returns:
        --------
        None
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        # default debug save location
        work_dir = os.path.dirname(__file__)
        debug_dir = os.path.join(work_dir, "plots_debug")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        msd = msd_list[0]
        n_sim = msd.n_sim
        n_mass = msd.n_mass

        if n_sim > 6:
            rng = np.random.default_rng()
            idx_plt_n_sim = rng.integers(0, n_sim, (6,))
        else:
            idx_plt_n_sim = np.arange(n_sim)

        linestyles = ["solid", "dashed", "dashdot"]
        legend = []
        fig, ax = plt.subplots(n_mass, 2)
        fig.suptitle(f"compare msd variants")
        for i_msd, msd in enumerate(msd_list):
            assert msd.n_sim == n_sim  # same as of first msd
            assert msd.n_mass == n_mass  # same as of first msd
            n_sim, n_t, n_n, n_dn = msd.X.shape
            X = np.zeros_like(msd.X)
            if msd.system_type == "ph":
                if msd.Q_is_identity:
                    # transform into state-space coordinates
                    # X = np.reshape(msd.X, (n_sim, msd.X.shape[1], n_mass * 2), "F")
                    for i_sim in range(n_sim):
                        x_i = reshape_states_to_features(
                            msd.X[i_sim, :, :, :][np.newaxis, :]
                        )
                        x_i_T = np.transpose(msd.T_inv[:, :, i_sim] @ x_i.T)
                        X[i_sim, :, :, :] = reshape_features_to_states(
                            x_i_T, n_sim=1, n_t=n_t, n_n=n_n, n_dn=n_dn
                        )
                else:
                    X = msd.X
            elif msd.system_type == "ss":
                for i_sim in range(n_sim):
                    # convert to momentum
                    x_i = reshape_states_to_features(
                        msd.X[i_sim, :, :, :][np.newaxis, :]
                    )
                    x_i_T = np.transpose(
                        linalg.block_diag(np.eye(msd.M.shape[0]), msd.M[:, :, i_sim])
                        @ x_i.T
                    )
                    X[i_sim, :, :, :] = reshape_features_to_states(
                        x_i_T, n_sim=1, n_t=n_t, n_n=n_n, n_dn=n_dn
                    )
            else:
                raise NotImplementedError

            for i_mass in range(n_mass):
                ax[i_mass, 0].plot(
                    msd.t, X[idx_plt_n_sim, :, i_mass, 0].T, linestyle=linestyles[i_msd]
                )
                ax[i_mass, 1].plot(
                    msd.t, X[idx_plt_n_sim, :, i_mass, 1].T, linestyle=linestyles[i_msd]
                )
            legend_str = f"{msd.system_type}_Qid{msd.Q_is_identity}"
            legend.append(legend_str)
        plt.legend(legend)
        plt.xlabel("time [s]")
        plt.show(block=False)
        plt.savefig(os.path.join(debug_dir, "msd_comparison.png"))
