import os
import logging
import numpy as np
from scipy import linalg
from scipy.stats import qmc
from scipy import signal


from matplotlib import pyplot as plt
import matplotlib as mpl


class ParameterInput:
    def __init__(
        self,
        n_mass,
        mass_vals,
        stiff_vals,
        damp_vals,
        input_vals,
        u,
        x0,
        parameter_information,
        Mu_input: np.ndarray = None,
    ) -> None:
        """
        Initialize a ParameterInput object with the given parameters.

        Parameters:
        - n_mass (int): Number of masses in the system.
        - mass_vals (np.ndarray): Array of mass values with shape (n_mass, n_sim).
        - stiff_vals (np.ndarray): Array of stiffness values with shape (n_mass, n_sim).
        - damp_vals (np.ndarray): Array of damping values with shape (n_mass, n_sim).
        - input_vals (array-like or None): Indices of excited masses for input.
        - u (np.ndarray): Input functions as lambda functions.
        - x0 (np.ndarray): Initial condition array with shape (2 * n_mass, n_sim).
        - parameter_information (str): Description of parameter space.
        """
        self.n_mass = n_mass
        self.n_sim = mass_vals.shape[1]
        self.parameter_information = parameter_information

        # system parameters
        assert mass_vals.ndim == 2 and mass_vals.shape[0] == n_mass
        assert (
            stiff_vals.ndim == 2
            and stiff_vals.shape[0] == n_mass
            and stiff_vals.shape[1] == self.n_sim
        )
        assert (
            damp_vals.ndim == 2
            and damp_vals.shape[0] == n_mass
            and damp_vals.shape[1] == self.n_sim
        )
        assert (
            x0.ndim == 2
            and x0.shape[0] == 2 * self.n_mass
            and x0.shape[1] == self.n_sim
        )
        self.mass_vals = mass_vals
        self.stiff_vals = stiff_vals
        self.damp_vals = damp_vals
        # input parameters
        self.input_vals = input_vals
        self.u = u
        self.Mu_input = Mu_input
        # initial condition
        self.x0 = x0

    @classmethod
    def from_config(cls, config, seed=1):
        """
        Create a ParameterInput object from a configuration dictionary.

        Parameters:
        - config (dict): Configuration dictionary containing system parameters and settings.
        - seed (int, default=1): Random seed for sampling and initialization.
        """
        parameter_method = config["parameter_method"]
        n_mass = config["n_mass"]
        parameter_information = "parameter_space = [mass stiffness damping omega delta]"

        # parameter with intervalls are sampled using Halton sequence
        param_config = config[config["parameter_method"]]
        n_simulations_per_parameter_set = param_config[
            "n_simulations_per_parameter_set"
        ]
        mass_vals = param_config["mass_vals"]
        stiff_vals = param_config["stiff_vals"]
        damp_vals = param_config["damp_vals"]
        delta = param_config["delta"]
        omega = param_config["omega"]

        params = [mass_vals, stiff_vals, damp_vals, omega, delta]
        lower_bounds = [limit[0] if isinstance(limit, list) else 0 for limit in params]
        upper_bounds = [limit[1] if isinstance(limit, list) else 10 for limit in params]
        random_samples = config["random_samples"]
        n_param_sets = int(random_samples / n_simulations_per_parameter_set)
        parameter_dimension = 5
        sampler_Halton = qmc.Halton(d=parameter_dimension, seed=seed)
        samples_Halton = sampler_Halton.random(n=random_samples)
        sampled_parameters = qmc.scale(samples_Halton, lower_bounds, upper_bounds)

        # repeat parameters n_simulations_per_parameter_set times
        mass_vals = (
            np.repeat(
                sampled_parameters[:n_param_sets, 0], n_simulations_per_parameter_set
            )
            if isinstance(mass_vals, list)
            else mass_vals
        )
        stiff_vals = (
            np.repeat(
                sampled_parameters[:n_param_sets, 1], n_simulations_per_parameter_set
            )
            if isinstance(stiff_vals, list)
            else stiff_vals
        )
        damp_vals = (
            np.repeat(
                sampled_parameters[:n_param_sets, 2], n_simulations_per_parameter_set
            )
            if isinstance(damp_vals, list)
            else damp_vals
        )
        omega = sampled_parameters[:, 3] if isinstance(omega, list) else omega
        delta = sampled_parameters[:, 4] if isinstance(delta, list) else delta

        Mu_input = np.concatenate(
            (np.expand_dims(omega, axis=1), np.expand_dims(delta, axis=1)), axis=1
        )

        input_vals = config["input_vals"]
        u, n_sim = cls.process_config_input_parameters(
            delta,
            omega,
            input_vals,
            n_sim=config["n_mass"],
            parameter_method=parameter_method,
        )
        mass_vals = cls.process_config_system_parameters(mass_vals, n_mass, n_sim)
        stiff_vals = cls.process_config_system_parameters(stiff_vals, n_mass, n_sim)
        damp_vals = cls.process_config_system_parameters(damp_vals, n_mass, n_sim)

        if config["debug"]:
            if input_vals is not None:
                plot_input(u, config)

        # generate initial condition
        x0 = cls.generate_initial_condition(
            config["random_initial_condition"], 2 * n_mass, n_sim, seed=seed
        )

        return cls(
            n_mass,
            mass_vals,
            stiff_vals,
            damp_vals,
            input_vals,
            u,
            x0,
            parameter_information,
            Mu_input,
        )

    @staticmethod
    def generate_initial_condition(random, n, n_sim, seed=1):
        """
        Generate initial conditions for the system.

        Parameters:
        - input_vals (array-like or None): Input values that influence the initial condition.
        - n (int): Total number of states (2 * n_mass).
        - n_sim (int): Number of simulations.
        - seed (int, default=1): Random seed for generating initial conditions.

        Returns:
        - np.ndarray: Array of initial conditions with shape (n, n_sim).
        """
        rng = np.random.default_rng(seed=seed)
        if random:
            # random initial conditions
            x0 = (rng.random((n, n_sim)) - 0.5) * 0.01
            # set velocities to zero
            x0[3:] = x0[3:] * 0
        else:
            # zero initial condition if input is used
            x0 = np.zeros((n, n_sim))
        return x0

    def process_config_system_parameters(config_parameter, n_mass, n_sim):
        """
        Process and format system parameters from the configuration.

        Parameters:
        - config_parameter (int, list, or np.ndarray): System parameter values.
        - n_mass (int): Number of masses.
        - n_sim (int): Number of simulations.

        Returns:
        - np.ndarray: Array of system parameters with shape (n_mass, n_sim).
        """
        if isinstance(config_parameter, int) or isinstance(config_parameter, float):
            # (int) use same value for masses
            config_vals = np.ones((n_mass, n_sim)) * config_parameter
        elif isinstance(config_parameter, list):
            # (list) needs to coincide with n_mass - sets system value for each mass separately
            assert len(config_parameter) == n_mass
            config_parameter = np.array(config_parameter)[:, np.newaxis]
            config_vals = np.repeat(np.array(config_parameter), n_sim, axis=1)
        elif isinstance(config_parameter, np.ndarray):
            if config_parameter.ndim == 1:
                # (1D-array) needs to coincide with n_sim - sets same system value for each mass
                assert config_parameter.shape[0] == n_sim
                config_vals = np.repeat(
                    np.array(config_parameter)[np.newaxis, :], n_mass, axis=0
                )
            elif config_parameter.ndim == 2:
                # (2D-array) needs to coincide with n_mass - sets system value for each mass separately
                assert config_parameter.shape[0] == n_mass
                assert config_parameter.shape[1] == n_sim
                config_vals = config_parameter
        return config_vals

    def process_config_input_parameters(
        delta, omega, input_vals, n_sim=None, parameter_method=""
    ):
        """
        Process and generate input parameters based on the configuration.

        Parameters:
        - delta (float, list, np.ndarray, or None): Damping values.
        - omega (float, list, np.ndarray, or None): Frequency values.
        - input_vals (array-like or None): Indices for input excitation.
        - n_sim (int, optional): Number of simulations.
        - parameter_method (str, optional): Method used for generating parameters ("default", "manual", or "Halton").

        Returns:
        - u (np.ndarray): Array of input functions as lambda functions.
        - n_sim (int): Number of simulations.
        """
        if input_vals is None:
            # no input - autonomous system
            if delta is not None or omega is not None:
                logging.warning(
                    f"Autonomous case. Values for delta and omega are not used. Set input_vals to excitation index if an input is desired."
                )
            u = None
            # use n_sim from config
        else:
            if isinstance(input_vals, int):
                # single mass excitation (SISO)
                n_u = 1
            else:
                n_u = len(input_vals)
            standard_u = lambda t, delta, omega: np.exp(-delta * t) * np.sin(
                omega * t**2
            )  # in publication [MorandinNicodemusUnger22] with input u(t) = exp(-t/2)*sin(t^2)
            if parameter_method == "default":
                # special case with sawtooth
                u = np.array(
                    [
                        lambda t: np.ones(
                            n_u,
                        )
                        * standard_u(t, omega, delta),
                        lambda t: np.ones(
                            n_u,
                        )
                        * signal.sawtooth(2 * np.pi * 0.5 * t),
                    ]
                )
                n_sim = 2
            else:
                # get input from delta and omega
                if (isinstance(omega, int) or isinstance(omega, float)) and (
                    isinstance(delta, int) or isinstance(delta, float)
                ):
                    u = np.array(
                        [
                            lambda t: np.ones(
                                n_u,
                            )
                            * standard_u(t, delta, omega)
                        ]
                    )
                    n_sim = 1
                elif isinstance(omega, list) and isinstance(delta, list):
                    # multiple simulations
                    assert len(omega) == len(delta)
                    u_list = []
                    for i in range(len(omega)):
                        u_list.append(
                            lambda t, i=i: np.ones(
                                n_u,
                            )
                            * standard_u(t, delta[i], omega[i])
                        )
                    u = np.array(u_list)
                    n_sim = len(omega)
                elif isinstance(omega, np.ndarray) and isinstance(delta, np.ndarray):
                    # multiple simulations
                    if omega.ndim == 1:
                        assert omega.shape[0] == delta.shape[0]
                        u_list = []
                        for i in range(omega.shape[0]):
                            u_list.append(
                                lambda t, i=i: np.ones(
                                    n_u,
                                )
                                * standard_u(t, delta[i], omega[i])
                            )
                        u = np.array(u_list)
                        n_sim = omega.shape[0]
                    else:
                        raise ValueError(f"omega has wrong dimension.")
                else:
                    raise ValueError(f"omega and delta have the wrong format")

        return u, n_sim

    # def create_input(omega, delta, siso, config, debug=False):

    # if siso:
    #     n_trajectories = 1
    #     u = lambda t: np.exp(-delta * t) * np.sin(
    #         omega * t**2
    #     )  # in publication [MorandinNicodemusUnger22] with input u(t) = exp(-t/2)*sin(t^2)
    # else:
    #     n_trajectories = 4
    #     amplitude = 1
    #     u_mult = lambda t, delta, omega: np.exp(-delta * t) * amplitude * np.sin(omega * t**2)
    #     u_mult_1 = lambda t: u_mult(t, 1, 1)  # standard case
    #     u_mult_2 = lambda t: u_mult(t, 2, 1)  # higher amp
    #     u_mult_3 = lambda t: u_mult(t, 1, 2)  # higher freq
    #     u_mult_4 = lambda t: u_mult(t, 2, 2)  # higher amp and freq
    #     u = [u_mult_1, u_mult_2, u_mult_3, u_mult_4]

    # u_test = lambda t: signal.sawtooth(2 * np.pi * 0.5 * t)

    # if debug:
    #     plot_input(u, u_test, config)


def plot_input(u, config, num_plot_max: int = 6):
    mpl.rcParams.update(mpl.rcParamsDefault)
    # default debug save location
    work_dir = os.path.dirname(__file__)
    debug_dir = os.path.join(work_dir, "plots_debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # time values for trajectories
    T_training = config["T_training"]
    time_steps_training = config["time_steps_training"]
    # T_test = config["T_test"]
    # time_steps_test = config["time_steps_test"]
    t_training = np.linspace(0, T_training, time_steps_training)
    # t_test = np.linspace(0, T_test, time_steps_test)

    n_u = u.shape[0]
    if n_u > num_plot_max:
        logging.info(f"More than {num_plot_max} inputs. Not plotting all of them.")
        n_u = num_plot_max
    fig, ax = plt.subplots(n_u)
    for i in range(n_u):
        u_i = u[i]
        plt.plot(t_training, u_i(t_training))

    plt.title("input")
    plt.xlabel("time [s]")
    plt.show(block=False)
    plt.savefig(os.path.join(debug_dir, "input_plot.png"))
