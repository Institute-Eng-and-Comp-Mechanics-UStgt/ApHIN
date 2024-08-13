"""
Utilities for the use in the examples
"""

import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from aphin.layers import PHQLayer

import pandas as pd

import itertools
import random

# logging setup
logging.basicConfig()


def setup_matplotlib(save_plots=False):
    """
    Set up matplotlib for generating plots, with an option to save them as PGF files.

    This function configures matplotlib to produce high-quality plots with LaTeX formatting.
    If `save_plots` is set to `True`, the plots will be saved directly as PGF files, and the
    necessary directories will be created if they do not exist.

    Parameters:
    -----------
    save_plots : bool, optional
        If `True`, plots will be saved directly to PGF files in the "results" directory instead of being shown.
        Defaults to `False`.

    Notes:
    ------
    - When `save_plots` is `True`, matplotlib is configured to use PGF backend for creating plots, which
      are suitable for LaTeX documents.
    - The function also sets up the LaTeX preamble to include packages like `amsmath` and `bm` for advanced
      mathematical typesetting.
    - The default settings ensure that the font used is "Computer Modern Roman" with a font size of 11, and
      labels on the axes are large.
    - The function updates `rcParams` multiple times to apply the desired settings for either saving or displaying plots.
    """
    if save_plots:
        if not os.path.exists("results"):
            os.makedirs("results")
        matplotlib.use("pgf")
        plt.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "pgf.rcfonts": False,
                "pgf.preamble": "\n".join(
                    [
                        r"\usepackage{amsmath}",
                        r"\usepackage{bm}",
                    ]
                ),
            }
        )
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{bm}",
                ]
            ),
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            "font.size": 11,
            "axes.labelsize": "large",
        }
    )
    plt.rcParams.update(
        {
            "pgf.texsystem": "lualatex",
            "pgf.rcfonts": False,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{bm}",
                ]
            ),
        }
    )
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{bm}",
                ]
            ),
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            "font.size": 11,
            "axes.labelsize": "large",
        }
    )


def animate_parameter_sweep(
    ph_layer, mu, mu_names, param_id=0, directory="results", save=False
):
    """
    Creates an animation of the parameter sweep for system matrices and saves it as a GIF.

    This function generates an animation that visualizes the effect of varying a specific parameter
    on the system matrices \( J \), \( R \), \( B \), and \( Q \). The animation is created by sweeping
    through a range of values for the selected parameter and updating the plot accordingly.

    Parameters:
    -----------
    ph_layer : object
        The layer object responsible for computing the system matrices. It should have a method
        `get_system_matrices` that returns the matrices based on the input parameters.
    mu : numpy.ndarray
        An array of parameters used to compute the system matrices. It should have dimensions
        (n_parameters, n_samples).
    mu_names : list of str
        A list of names for the parameters, where each name corresponds to a column in `mu`.
    param_id : int, optional
        The index of the parameter to sweep through. Defaults to 0.
    directory : str, optional
        Directory where the animation GIF will be saved if `save` is `True`. Defaults to "results".
    save : bool, optional
        If `True`, saves the animation as a GIF in the specified `directory`. Defaults to `False`.
    """
    n_mu = mu.shape[1]
    param_name = mu_names[param_id]
    parameter_limits = [mu.min(axis=0), mu.max(axis=0)]
    param_id = 2  # 0: m, 1: k, 2: c
    param_sweep = np.zeros((n_mu, 1))
    param_sweep[param_id] = 1
    param_sweep = (
        param_sweep
        * np.linspace(parameter_limits[0][param_id], parameter_limits[1][param_id], 100)
    ).T
    # predict matrices for each parameter
    if ph_layer is PHQLayer:
        J_pred, R_pred, B_pred, Q_pred = ph_layer.get_system_matrices(param_sweep)
    else:
        J_pred, R_pred, B_pred = ph_layer.get_system_matrices(param_sweep)
        Q_pred = np.zeros_like(J_pred)

    # plot comparison of original and reconstructed data as animation
    min_values = [J_pred.min(), R_pred.min(), B_pred.min(), Q_pred.min()]
    max_values = [J_pred.max(), R_pred.max(), B_pred.max(), Q_pred.max()]

    fig, ax = plt.subplots(1, 4, figsize=(12, 3), dpi=300, sharex="all", sharey="all")
    # global title
    title = fig.suptitle(
        f"Parameter sweep over ${param_name}$ with value {param_sweep[0, param_id]: .2f}"
    )
    # imshow matrices
    im1 = ax[0].imshow(J_pred[0], vmin=min_values[0], vmax=max_values[0])
    ax[0].set_title("$J_{ph}$")
    im2 = ax[1].imshow(R_pred[0], vmin=min_values[1], vmax=max_values[1])
    ax[1].set_title("$R_{ph}$")
    im3 = ax[2].imshow(B_pred[0], vmin=min_values[2], vmax=max_values[2])
    ax[2].set_title("$B_{ph}$")
    im4 = ax[3].imshow(Q_pred[0], vmin=min_values[3], vmax=max_values[3])
    ax[3].set_title("$Q_{ph}$")
    fig.tight_layout()

    # initialization function: plot the background of each frame
    def init():
        title.set_text(
            f"Parameter sweep over ${param_name}$ with value {param_sweep[0, param_id]: .2f}"
        )
        im1.set_data(J_pred[0, :, :])
        im2.set_data(R_pred[0, :, :])
        im3.set_data(B_pred[0, :, :])
        im4.set_data(Q_pred[0, :, :])
        return []  # , im2, im3

    # animation function.  This is called sequentially
    def animate(i):  # exponential decay of the values
        # print(i)
        title.set_text(
            f"Parameter sweep over ${param_name}$ with value {param_sweep[i, param_id]: .2f}"
        )
        im1.set_array(J_pred[i, :, :])
        im2.set_array(R_pred[i, :, :])
        im3.set_array(B_pred[i, :, :])
        im4.set_array(Q_pred[i, :, :])
        return []  # , im2, im3

    ani = animation.FuncAnimation(
        fig, animate, frames=range(0, 100), init_func=init, blit=True
    )

    # save animation as gif
    # To save the animation using Pillow as a gif
    if save:
        if not os.path.exists(directory):
            logging.info(f"Creating directory {os.path.abspath(directory)}")
            os.makedirs(directory)
        save_path = os.path.join(directory, f"parameter_sweep_{param_name}.gif")
        writer = animation.PillowWriter(
            fps=15, metadata=dict(artist="Me"), bitrate=1800
        )
        ani.save(save_path, writer=writer)


def plot_X_comparison(
    original_data, identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots a comparison of state results between original and identified data.
    Identified state means that the results are time integrated in the latent space and decoded.

    This function generates plots comparing the state variables of the original and identified
    data instances. The comparison is made using randomly selected or sequentially generated
    indices for the state variables, based on the specified `idx_gen` method.

    Parameters:
    -----------
    original_data : object
        The instance of the original data containing the true state results.
    identified_data : object
        The instance of the identified data containing the identified state results.
    use_train_data : bool, optional
        If `True`, use training data for plotting. If `False`, use test data. Defaults to `False`.
    idx_gen : str, optional
        Method for generating indices. Options are `"rand"` for random indices or `"first"` for sequential indices. Defaults to `"rand"`.
    save_path : str, optional
        Directory path to save the plot image. If empty, the plot will not be saved. Defaults to `""`.

    Returns:
    -----------
    None
    """

    t, X, X_id, idx_n_n, idx_n_dn, idx_sim, _, num_plots = get_quantity_of_interest(
        original_data, identified_data, "X", "X", use_train_data, "X", idx_gen
    )

    variable_names = [r"\bm{X}", r"\bm{X}_\mathrm{ph}"]
    save_name = "X"
    plot_X(
        num_plots,
        t,
        X,
        X_id,
        idx_n_n,
        idx_n_dn,
        idx_sim,
        variable_names,
        save_name=save_name,
        save_path=save_path,
    )


def plot_x_comparison(
    original_data, identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots a comparison of the original latent features with their identified counterparts.
    Identified state means that the results are time integrated in the latent space and decoded.

    This function generates plots to compare the original latent features (`x`) with their corresponding
    identified or predicted features (`x_id`). It uses either training or test data based on the `use_train_data`
    parameter, and the indices of features to be plotted can be chosen either randomly or sequentially.

    Parameters:
    -----------
    original_data : object
        The dataset containing the original latent features. This object should have attributes `TRAIN` and `TEST`
        representing the training and testing data, respectively.

    identified_data : object
        The dataset containing the identified or predicted latent features. This object should also have attributes
        `TRAIN` and `TEST` similar to `original_data`.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data.
        Default is False.

    idx_gen : str, optional
        Method for generating indices of features to plot. Options are:
        - "rand": Randomly selects indices.
        - "first": Selects the first few indices.
        Default is "rand".

    save_path : str, optional
        The directory path where the plot will be saved. If not provided, the plot will not be saved.

    Returns:
    --------
    None
        The function does not return anything. It generates and displays the plot(s), and optionally saves them.
    """

    _, x, x_id, _, _, _, idx_n_f, num_plots = get_quantity_of_interest(
        original_data, identified_data, "x", "x", use_train_data, "x", idx_gen
    )

    variable_names = [r"\bm{x}", r"\bm{x}_\mathrm{ph}"]
    save_name = "x"
    plot_x(
        num_plots,
        x,
        x_id,
        idx_n_f,
        variable_names,
        save_name=save_name,
        save_path=save_path,
    )


def plot_X_dt_comparison(
    original_data, identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots a comparison of state derivatives between original and identified data.
    Identified state derivative means that the results are time integrated in the latent space and retransformed/decoded through automatic differentiation.

    This function generates plots comparing the state derivatives (i.e., time derivatives of the states)
    of the original and identified data instances. The comparison is made using randomly selected
    or sequentially generated indices for the state variables, based on the specified `idx_gen` method.

    Parameters:
    -----------
    original_data : object
        The instance of the original data containing the true state derivatives.
    identified_data : object
        The instance of the identified data containing the estimated state derivatives.
    use_train_data : bool, optional
        If `True`, use training data for plotting. If `False`, use test data. Defaults to `False`.
    idx_gen : str, optional
        Method for generating indices. Options are `"rand"` for random indices or `"first"` for sequential indices. Defaults to `"rand"`.
    save_path : str, optional
        Directory path to save the plot image. If empty, the plot will not be saved. Defaults to `""`.

    Returns:
    -----------
    None
    """

    t, X_dt, X_dt_id, idx_n_n, idx_n_dn, idx_sim, _, num_plots = (
        get_quantity_of_interest(
            original_data, identified_data, "X_dt", "X_dt", use_train_data, "X", idx_gen
        )
    )

    variable_names = [r"\dot{\bm{X}}", r"\dot{\bm{X}}_\mathrm{ph}"]
    save_name = "X_dt"
    plot_X(
        num_plots,
        t,
        X_dt,
        X_dt_id,
        idx_n_n,
        idx_n_dn,
        idx_sim,
        variable_names,
        save_name=save_name,
        save_path=save_path,
    )


def plot_x_reconstruction(
    original_data, identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots and compares the original and autoencoder-reconstructed time series data for multiple features. The data is only encoded and decoded, the pHIN layer is not involved.


    This function generates plots comparing the original time series data (`x`) with the reconstructed data (`x_rec`)
    produced by an autoencoder model. The function creates subplots for each feature, displaying both the original and
    reconstructed data. It saves the resulting figure as a PNG file if specified.

    Parameters:
    -----------
    original_data : object
        The dataset object containing the original data. Must have attributes to access time series data.

    identified_data : object
        The dataset object containing the autoencoder-reconstructed data. Must have attributes to access time series data.

    use_train_data : bool, optional
        If True, use the training data from `original_data` and `identified_data`; otherwise, use the test data. Default is False.

    idx_gen : str, optional
        Method for generating indices for the features to plot. Options are "rand" (random) or "first" (first N features). Default is "rand".

    save_path : str, optional
        Directory where the plot will be saved. If not provided, the plot will be saved in the current working directory.

    Returns:
    --------
    None
    """

    _, x, x_id, _, _, _, idx_n_f, num_plots = get_quantity_of_interest(
        original_data, identified_data, "x", "x_rec", use_train_data, "x", idx_gen
    )
    if x_id is not None:
        variable_names = [r"\bm{x}", r"\bm{x}_\mathrm{rec}"]
        save_name = r"x_rec"
        plot_x(
            num_plots,
            x,
            x_id,
            idx_n_f,
            variable_names,
            save_name=save_name,
            save_path=save_path,
        )


def plot_x_dt_reconstruction(
    original_data, identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots and compares the original and autoencoder-reconstructed time derivative of the data.
    The data is only encoded and decoded, the pHIN layer is not involved.

    This function generates plots comparing the original time derivative data (`dx_dt`) with the reconstructed data (`x_rec_dt`)
    produced by an autoencoder model. It creates subplots for each feature, displaying both the original and reconstructed time derivatives.
    The resulting figure can be saved as a PNG file if specified.

    Parameters:
    -----------
    original_data : object
        The dataset object containing the original time derivative data. Must have attributes to access time series data.

    identified_data : object
        The dataset object containing the autoencoder-reconstructed time derivative data. Must have attributes to access time series data.

    use_train_data : bool, optional
        If True, use the training data from `original_data` and `identified_data`; otherwise, use the test data. Default is False.

    idx_gen : str, optional
        Method for generating indices for the features to plot. Options are "rand" (random) or "first" (first N features). Default is "rand".

    save_path : str, optional
        Directory where the plot will be saved. If not provided, the plot will be saved in the current working directory.

    Returns:
    --------
    None
    """
    _, x_dt, x_dt_id, _, _, _, idx_n_f, num_plots = get_quantity_of_interest(
        original_data,
        identified_data,
        "dx_dt",
        "x_rec_dt",
        use_train_data,
        "x",
        idx_gen,
    )
    if x_dt_id is not None:
        variable_names = [r"\dot{\bm{x}}", r"\dot{\bm{x}}_{\mathrm{rec}}"]
        save_name = r"x_dt_rec"
        plot_x(
            num_plots,
            x_dt,
            x_dt_id,
            idx_n_f,
            variable_names,
            save_name=save_name,
            save_path=save_path,
        )


def plot_X_reconstruction(
    original_data, identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots a comparison of original states and reconstructed states from an autoencoder.
    The data is only encoded and decoded, the pHIN layer is not involved.

    This function generates plots comparing the original states with the states reconstructed
    by an autoencoder from the identified data. It selects a specified number of indices to plot
    based on the `idx_gen` method.

    Parameters:
    -----------
    original_data : object
        The instance of the original data containing the true states.
    identified_data : object
        The instance of the identified data containing the reconstructed states from the autoencoder.
    use_train_data : bool, optional
        If `True`, use training data for plotting. If `False`, use test data. Defaults to `False`.
    idx_gen : str, optional
        Method for generating indices. Options are `"rand"` for random indices or `"first"` for sequential indices. Defaults to `"rand"`.
    save_path : str, optional
        Directory path to save the plot image. If empty, the plot will not be saved. Defaults to `""`.

    Returns:
    -----------
    None
    """

    t, X, X_id, idx_n_n, idx_n_dn, idx_sim, _, num_plots = get_quantity_of_interest(
        original_data, identified_data, "X", "X_rec", use_train_data, "X", idx_gen
    )
    if X_id is not None:
        variable_names = [r"\bm{X}", r"\bm{X}_\mathrm{rec}"]
        save_name = "X_rec"
        plot_X(
            num_plots,
            t,
            X,
            X_id,
            idx_n_n,
            idx_n_dn,
            idx_sim,
            variable_names,
            save_name=save_name,
            save_path=save_path,
        )


def plot_X_dt_reconstruction(
    original_data, identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots a comparison of original state derivatives and reconstructed state derivatives from an autoencoder.
    The data is only encoded and decoded, the pHIN layer is not involved.

    This function generates plots comparing the original time derivatives of the states with the state derivatives
    reconstructed by an autoencoder from the identified data. It selects a specified number of indices for plotting
    based on the `idx_gen` method.

    Parameters:
    -----------
    original_data : object
        The instance of the original data containing the true state derivatives.
    identified_data : object
        The instance of the identified data containing the state derivatives reconstructed by the autoencoder.
    use_train_data : bool, optional
        If `True`, use training data for plotting. If `False`, use test data. Defaults to `False`.
    idx_gen : str, optional
        Method for generating indices. Options are `"rand"` for random indices or `"first"` for sequential indices. Defaults to `"rand"`.
    save_path : str, optional
        Directory path to save the plot image. If empty, the plot will not be saved. Defaults to `""`.

    Returns:
    -----------
    None
    """

    t, X, X_id, idx_n_n, idx_n_dn, idx_sim, _, num_plots = get_quantity_of_interest(
        original_data,
        identified_data,
        "X_dt",
        "X_rec_dt",
        use_train_data,
        "X",
        idx_gen,
    )
    if X_id is not None:
        variable_names = [r"\dot{\bm{X}}", r"\dot{\bm{X}}_{\mathrm{rec}}"]
        save_name = r"X_rec_dt"
        plot_X(
            num_plots,
            t,
            X,
            X_id,
            idx_n_n,
            idx_n_dn,
            idx_sim,
            variable_names,
            save_name=save_name,
            save_path=save_path,
        )


def plot_Z_ph(identified_data, use_train_data=False, idx_gen="rand", save_path=""):
    """
    Plots the comparison between identified latent variables (Z) and their corresponding port-Hamiltonian versions (Z_ph).
    where
        - Z: obtained from the encoded original data
        - Z_ph: obtained through the encoded initial condition of x0 and time integration with the identified system

    This function generates a series of plots that compare the latent variables `Z` with their port-Hamiltonian
    counterparts `Z_ph`, as computed by a model. The data can be selected from either the training or test set,
    and specific indices can be chosen randomly or in a sequential manner.

    Parameters:
    -----------
    identified_data : object
        The dataset containing the identified variables. This object should have attributes `TRAIN` and `TEST`
        representing the training and testing data, respectively.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data.
        Default is False.

    idx_gen : str, optional
        Method for generating indices of features to plot. Options are:
        - "rand": Randomly selects indices.
        - "first": Selects the first few indices.
        Default is "rand".

    save_path : str, optional
        The directory path where the plot will be saved. If not provided, the plot will not be saved.

    Returns:
    --------
    None
        The function does not return anything. It generates and displays the plot(s), and optionally saves them.
    """

    t, Z, Z_ph, _, _, idx_sim, idx_n_f, num_plots = get_quantities_of_interest(
        identified_data, "Z", "Z_ph", use_train_data, "Z", idx_gen
    )

    variable_names = [r"\bm{Z}", r"\bm{Z}_\mathrm{ph}"]
    save_name = "Z"
    plot_Z(
        num_plots,
        t,
        Z,
        Z_ph,
        idx_n_f,
        idx_sim,
        variable_names,
        save_name=save_name,
        save_path=save_path,
    )


def plot_z_ph(identified_data, use_train_data=False, idx_gen="rand", save_path=""):
    """
    Plots the comparison between identified latent variables (Z) and their corresponding port-Hamiltonian versions (Z_ph).
    where
        - Z: obtained from the encoded original data
        - Z_ph: obtained through the encoded initial condition of x0 and time integration with the identified system

    This function generates a series of plots that compare the latent features `z` with their port-Hamiltonian
    counterparts `z_ph`, as extracted from a dataset. The data can be selected from either the training or test set,
    and specific indices can be chosen either randomly or sequentially.

    Parameters:
    -----------
    identified_data : object
        The dataset containing the identified features. This object should have attributes `TRAIN` and `TEST`
        representing the training and testing data, respectively.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data.
        Default is False.

    idx_gen : str, optional
        Method for generating indices of features to plot. Options are:
        - "rand": Randomly selects indices.
        - "first": Selects the first few indices.
        Default is "rand".

    save_path : str, optional
        The directory path where the plot will be saved. If not provided, the plot will not be saved.

    Returns:
    --------
    None
        The function does not return anything. It generates and displays the plot(s), and optionally saves them.
    """
    _, z, z_ph, _, _, _, idx_n_f, num_plots = get_quantities_of_interest(
        identified_data, "z", "z_ph", use_train_data, "z", idx_gen
    )

    variable_names = [r"\bm{z}", r"\bm{z}_\mathrm{ph}"]
    save_name = "z"
    plot_z(
        num_plots,
        z,
        z_ph,
        idx_n_f,
        variable_names,
        save_name=save_name,
        save_path=save_path,
    )


def plot_Z_dt_ph_map(
    identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots the comparison between the time derivatives of the latent features (Z_dt) and their corresponding port-Hamiltonian mapped versions (Z_dt_ph_map).
    Where
        - Z_dt: obtained through automatic differentiation (chain rule) of the original data through the encoder
        - Z_dt_ph_map: obtained by inserting the encoded original state into the pH network, i.e. inserting z into the pH equation. z_dt=(J-R)Q + Bu (no time integration)

    This function generates a series of plots that compare the time derivatives of the latent features `Z_dt` with
    their port-Hamiltonian mapped counterparts `Z_dt_ph_map`, as extracted from a dataset. The data can be selected
    from either the training or test set, and specific indices can be chosen either randomly or sequentially.

    Parameters:
    -----------
    identified_data : object
        The dataset containing the identified features. This object should have attributes `TRAIN` and `TEST`
        representing the training and testing data, respectively.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data.
        Default is False.

    idx_gen : str, optional
        Method for generating indices of features to plot. Options are:
        - "rand": Randomly selects indices.
        - "first": Selects the first few indices.
        Default is "rand".

    save_path : str, optional
        The directory path where the plot will be saved. If not provided, the plot will not be saved.

    Returns:
    --------
    None
        The function does not return anything. It generates and displays the plot(s), and optionally saves them.
    """

    t, Z_dt, Z_dt_ph, idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots = (
        get_quantities_of_interest(
            identified_data, "Z_dt", "Z_dt_ph_map", use_train_data, "Z", idx_gen
        )
    )

    variable_names = [r"\dot{\bm{Z}}", r"\dot{\bm{Z}}_{\mathrm{phmap}}"]
    save_name = "Z_dt_ph_map"
    plot_Z(
        num_plots,
        t,
        Z_dt,
        Z_dt_ph,
        idx_n_f,
        idx_sim,
        variable_names,
        save_name=save_name,
        save_path=save_path,
    )


def plot_z_dt_ph_map(
    identified_data, use_train_data=False, idx_gen="rand", save_path=""
):
    """
    Plots a comparison of the derivative of latent features (`z_dt`) with their corresponding port-Hamiltonian variables
    (`z_dt_ph_map`) from the identified data.
    Where
        - z_dt: obtained through automatic differentiation (chain rule) of the original data through the encoder
        - z_dt_ph_map: obtained by inserting the encoded original state into the pH network, i.e. inserting z into the pH equation. z_dt=(J-R)Qz + Bu (no time integration)

    This function generates plots to compare the derivative of latent features (`z_dt`) with their
    corresponding phase map (`z_dt_ph_map`). The function uses either training or test data based on the
    `use_train_data` parameter. The indices of features to be plotted are chosen based on the `idx_gen`
    parameter, which can either be random or sequential.

    Parameters:
    -----------
    identified_data : object
        The dataset containing the identified or predicted latent features. This object should have attributes
        `TRAIN` and `TEST` representing the training and testing data, respectively.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data.
        Default is False.

    idx_gen : str, optional
        Method for generating indices of features to plot. Options are:
        - "rand": Randomly selects indices.
        - "first": Selects the first few indices.
        Default is "rand".

    save_path : str, optional
        The directory path where the plot will be saved. If not provided, the plot will not be saved.

    Returns:
    --------
    None
        The function does not return anything. It generates and displays the plot(s), and optionally saves them.
    """

    t, z, z_ph, _, _, _, idx_n_f, num_plots = get_quantities_of_interest(
        identified_data, "z_dt", "z_dt_ph_map", use_train_data, "z", idx_gen
    )

    variable_name = [r"\dot{\bm{z}}", r"\dot{\bm{z}}_{\mathrm{phmap}}"]
    save_name = "z_dt_ph_map"
    plot_z(
        num_plots,
        z,
        z_ph,
        idx_n_f,
        variable_name,
        save_name=save_name,
        save_path=save_path,
    )


def plot_Z_dt_ph(identified_data, use_train_data=False, idx_gen="rand", save_path=""):
    """
    Plots the comparison between the time derivatives of the reduced latent features (`Z_dt`) and their corresponding
    port-Hamiltonian versions (`Z_dt_ph`).
    Where
        - Z_dt: obtained through automatic differentiation (chain rule) of the original data through the encoder
        - Z_dt_ph: obtained through time integration of the identified system in the latent space and inserting Z into the pH system Z_dt = (J-R)QZ + Bu

    The function allows the selection of data from either the training or test set, and indices can be generated either randomly or sequentially.

    Parameters:
    -----------
    identified_data : object
        The dataset containing the identified features. This object should have attributes `TRAIN` and `TEST`
        representing the training and testing data, respectively.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data.
        Default is False.

    idx_gen : str, optional
        Method for generating indices of features to plot. Options are:
        - "rand": Randomly selects indices.
        - "first": Selects the first few indices.
        Default is "rand".

    save_path : str, optional
        The directory path where the plot will be saved. If not provided, the plot will not be saved.

    Returns:
    --------
    None
        The function does not return anything. It generates and displays the plot(s), and optionally saves them.
    """
    t, Z_dt, Z_dt_ph, _, _, idx_sim, idx_n_f, num_plots = get_quantities_of_interest(
        identified_data, "Z_dt", "Z_dt_ph", use_train_data, "Z", idx_gen
    )

    variable_names = [r"\dot{\bm{Z}}", r"\dot{\bm{Z}}_\mathrm{ph}"]
    save_name = "Z_dt"
    plot_Z(
        num_plots,
        t,
        Z_dt,
        Z_dt_ph,
        idx_n_f,
        idx_sim,
        variable_names,
        save_name=save_name,
        save_path=save_path,
    )


def plot_z_dt_ph(identified_data, use_train_data=False, idx_gen="rand", save_path=""):
    """
    Plots the comparison between the time derivatives of the reduced latent features (`z_dt`) and their corresponding
    port-Hamiltonian versions (`z_dt_ph`).
    Where
        - z_dt: obtained through automatic differentiation (chain rule) of the original data through the encoder
        - z_dt_ph: obtained through time integration of the identified system in the latent space and inserting z into the pH system z_dt = (J-R)Qz + Bu

    The function allows the selection of data from either the training or test set, and indices can be generated either randomly or sequentially.

    Parameters:
    -----------
    identified_data : object
        The dataset containing the identified features. This object should have attributes `TRAIN` and `TEST`
        representing the training and testing data, respectively.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data.
        Default is False.

    idx_gen : str, optional
        Method for generating indices of features to plot. Options are:
        - "rand": Randomly selects indices.
        - "first": Selects the first few indices.
        Default is "rand".

    save_path : str, optional
        The directory path where the plot will be saved. If not provided, the plot will not be saved.

    Returns:
    --------
    None
        The function does not return anything. It generates and displays the plot(s), and optionally saves them.
    """
    _, z_dt, z_dt_ph, _, _, _, idx_n_f, num_plots = get_quantities_of_interest(
        identified_data, "z_dt", "z_dt_ph", use_train_data, "z", idx_gen
    )

    plot_z(
        num_plots,
        z_dt,
        z_dt_ph,
        idx_n_f,
        [r"\dot{\bm{z}}", r"\dot{\bm{z}}_\mathrm{ph}"],
        save_name="z_dt",
        save_path=save_path,
    )


def get_quantities_of_interest(
    data,
    quantity_1: str,
    quantity_2: str,
    use_train_data=False,
    data_type="X",
    idx_gen="rand",
):
    """
    Extracts and returns selected quantities of interest from a dataset for further analysis.

    This function retrieves two specified quantities (e.g., state variables, time derivatives, etc.) from either the training or test
    data of a given dataset object. It also generates indices for simulation, nodes, and features, which can be used for subsequent
    plotting or analysis tasks.

    Parameters:
    -----------
    data : object
        The dataset object containing the time series data. The object should have attributes for accessing the training and test data,
        as well as the time variable (`t`).

    quantity_1 : str
        The name of the first quantity to retrieve from the dataset object (e.g., "X", "dx_dt", etc.).

    quantity_2 : str
        The name of the second quantity to retrieve from the dataset object. This can be another state variable or a reconstructed quantity.

    use_train_data : bool, optional
        If True, the function retrieves the data from the training dataset; otherwise, it uses the test dataset. Default is False.

    data_type : str, optional
        The type of data being analyzed, such as "X" for state variables or "Z" for latent variables. Default is "X".

    idx_gen : str, optional
        Method for generating indices for selecting nodes, degrees of freedom, or features. Options are "rand" (random selection)
        or "first" (select the first N features). Default is "rand".

    Returns:
    --------
    tuple
        A tuple containing the following elements:
        - t (array-like): The time variable from the dataset.
        - quantity_1 (array-like): The first quantity of interest from the dataset.
        - quantity_2 (array-like): The second quantity of interest from the dataset.
        - idx_n_n (array-like): Indices for the nodes.
        - idx_n_dn (array-like): Indices for the degrees of freedom.
        - idx_sim (int): The index of the selected simulation.
        - idx_n_f (array-like): Indices for the selected features.
        - num_plots (int): The number of plots to be generated.
    """
    if use_train_data:
        data = data.TRAIN
    else:
        data = data.TEST

    idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots = get_sim_idx(
        data, data_type=data_type, idx_gen=idx_gen
    )

    t = data.t
    quantity_1 = getattr(data, quantity_1)
    quantity_2 = getattr(data, quantity_2)

    return t, quantity_1, quantity_2, idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots


def get_quantity_of_interest(
    original_data,
    identified_data,
    og_quantity: str,
    id_quantity: str,
    use_train_data=False,
    data_type="X",
    idx_gen="rand",
):
    """
    Retrieves and prepares quantities of interest for comparison between original and identified data.

    This function extracts specified quantities from both the original and identified datasets,
    based on the type of data (training or test) and the indices generation method. It provides the
    necessary time vector, quantities for comparison, and indices for plotting.

    Parameters:
    -----------
    original_data : object
        Dataset containing the original state variables. Should have attributes for training and testing data.
    identified_data : object
        Dataset containing the identified state variables. Should have attributes for training and testing data.
    og_quantity : str
        The name of the quantity to retrieve from the original data.
    id_quantity : str
        The name of the quantity to retrieve from the identified data.
    use_train_data : bool, optional
        If `True`, the training data will be used from both datasets. If `False`, the test data will be used. Defaults to `False`.
    data_type : str, optional
        Specifies the type of data to be retrieved. Defaults to "X".
    idx_gen : str, optional
        Method for generating indices. Defaults to "rand" for random indices.

    Returns:
    -----------
    t : array-like
        Time vector from the identified data.
    quantity_1 : array-like
        Quantity retrieved from the original data.
    quantity_2 : array-like
        Quantity retrieved from the identified data.
    idx_n_n : list
        Indices for the nodes variables.
    idx_n_dn : list
        Indices for the dof per node variables.
    idx_sim : int
        Index for simulation selection.
    idx_n_f : list
        Indices for the feature variables.
    num_plots : int
        Number of plots to be generated.
    """
    if use_train_data:
        original_data = original_data.TRAIN
        identified_data = identified_data.TRAIN
    else:
        original_data = original_data.TEST
        identified_data = identified_data.TEST

    idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots = get_sim_idx(
        original_data, data_type=data_type, idx_gen=idx_gen
    )

    t = identified_data.t
    quantity_1 = getattr(original_data, og_quantity)
    quantity_2 = getattr(identified_data, id_quantity)

    for quantity in [quantity_1, quantity_2]:
        if quantity is None:
            logging.info(
                f"Quantity {quantity} is not given in the dataset. Plot will not be created."
            )

    return t, quantity_1, quantity_2, idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots


def plot_errors(
    data,
    use_train_data=False,
    t=None,
    # title_label="",
    save_name="rms_error",
    domain_names=None,
    save_to_csv=False,
    yscale="linear",
):
    """
    Generates and saves plots of RMS errors for state and latent errors from the given dataset.

    This function plots RMS errors for different domains and latent features. The plots can be saved as PNG files and
    optionally as CSV files. The function handles multiple domains by iterating over them and calling `single_error_plot`
    for each domain's state error. It also plots the latent error if available.

    Parameters:
    -----------
    data : object
        The dataset containing error information. This object should have attributes `TRAIN` and `TEST` representing the
        training and testing data, respectively, and `state_error_list` and `latent_error` containing the RMS errors.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data. Default is False.

    t : numpy.ndarray or None, optional
        A 1D array representing time points corresponding to the error data. If None, defaults to the range of indices for
        the time points. Default is None.

    save_name : str, optional
        The base name for the files to which the plots will be saved. The name will be prefixed with "state_" for state
        errors and "latent_" for latent errors. Default is "rms_error".

    domain_names : list of str or None, optional
        A list of domain names corresponding to the state errors. If None, domain names will be generated automatically
        based on the number of state error domains. The length of the list must match the length of `data.state_error_list`.
        Default is None.

    save_to_csv : bool, optional
        If True, the error data will be saved to a CSV file. Default is False.

    yscale : str, optional
        The scale of the y-axis for the plots. Options are "linear" or "log". Default is "linear".

    Returns:
    --------
    None
        The function does not return anything. It generates and displays plots, and optionally saves them to files.
    """
    if use_train_data:
        data = data.TRAIN
    else:
        data = data.TEST

    # plot state error
    if domain_names is None:
        if len(data.state_error_list) > 1:
            domain_names = [f"dom{i}" for i in range(len(data.state_error_list))]
        else:
            # just one domain - do not distinguish
            domain_names = [""]
    else:
        # domain_names must be of same length as state_error_list which is equivalent to len(domains_split_vals)
        assert len(domain_names) == len(data.state_error_list)

    for i_domain, norm_rms_error in enumerate(data.state_error_list):
        save_name_dom = f"{save_name}_state_{domain_names[i_domain]}"
        title_label = f"state_error_{domain_names[i_domain]}"
        single_error_plot(
            norm_rms_error=norm_rms_error,
            t=t,
            title_label=title_label,
            save_name=save_name_dom,
            save_to_csv=save_to_csv,
            yscale=yscale,
        )

    # plot latent error
    if data.latent_error is not None:
        save_name_lat = f"{save_name}_latent"
        title_label = f"latent_error"
        single_error_plot(
            norm_rms_error=data.latent_error,
            t=t,
            title_label=title_label,
            save_name=save_name_lat,
            save_to_csv=save_to_csv,
            yscale=yscale,
        )


def single_error_plot(
    norm_rms_error,
    t=None,
    title_label="",
    save_name="rms_error",
    save_to_csv=False,
    yscale="linear",
):
    """
    Generates a plot of RMS error across simulations and optionally saves it to a PNG file and/or a CSV file.

    This function plots the root mean square (RMS) error values over time for a set of simulations. It computes
    the mean RMS error across all simulations and plots both the individual simulation errors and the mean error.
    The plot can be customized with a title, and the y-axis scale can be adjusted. Optionally, the data can be saved
    to a CSV file.

    Parameters:
    -----------
    norm_rms_error : numpy.ndarray
        A 2D array where each row represents RMS errors from a different simulation, and each column represents
        errors at a specific time point. Shape should be (n_simulations, n_time_points).

    t : numpy.ndarray or None, optional
        A 1D array representing time points corresponding to the columns of `norm_rms_error`. If None, default
        to the range of indices for the time points. Default is None.

    title_label : str, optional
        A string to be appended to the plot title to provide additional context or labeling. Default is an empty string.

    save_name : str, optional
        The name of the file (without extension) to which the plot will be saved as a PNG. Default is "rms_error".

    save_to_csv : bool, optional
        If True, the RMS error data along with the mean error will be saved to a CSV file. Default is False.

    yscale : str, optional
        The scale of the y-axis. Options are "linear" or "log". Default is "linear".

    Returns:
    --------
    None
        The function does not return anything. It generates and displays the plot(s), and optionally saves them
        as a PNG and/or CSV file.
    """

    # calculate mean value over all simulations
    mean_norm_rms_error = np.mean(norm_rms_error, axis=0)

    if t is None:
        t = range(norm_rms_error.shape[1])

    plt.figure()
    plt.plot(t, np.transpose(norm_rms_error), "gray", alpha=0.6, label="e_sim")
    plt.plot(t, np.transpose(mean_norm_rms_error), "r", label="e_mean")
    plt.yscale(yscale)
    plt.title(f"RMS error {title_label}")
    # remove duplicate labels through dictionary
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    save_as_png(save_name)
    plt.show(block=False)

    if save_to_csv:
        # if there are too many data points, pgfplots will throw an error and be very slow
        # limit data points to 100 for each trajectory
        data_point_limit = 100
        if t.shape[0] > data_point_limit:
            data_point_stepping = int(t.shape[0] / data_point_limit)
            t = t[::data_point_stepping]
            mean_norm_rms_error = mean_norm_rms_error[::data_point_stepping]
            norm_rms_error = norm_rms_error[:, ::data_point_stepping]
        # concatenate data
        data_array = np.concatenate(
            (t, np.transpose(norm_rms_error), mean_norm_rms_error[:, np.newaxis]),
            axis=1,
        )
        # create header
        header_list = ["t"]
        for i in range(norm_rms_error.shape[0]):
            header_list.append(f"error_{title_label}_{i}")
        header_list.append(f"mean_error_{title_label}")
        data_pd = pd.DataFrame(data_array)
        data_pd.to_csv(f"{save_name}.csv", header=header_list, index=False)


def get_sim_idx(data_instance, data_type="X", num_plots_max=6, idx_gen="rand"):
    """
    Generates indices for plotting based on the type of data and the number of plots desired.

    This function determines which indices to use for plotting the results based on the data type
    (`"X"` for state results or `"Z"` for latent results), the maximum number of plots, and the
    method for index generation. It provides indices for states or latent variables, as well as a
    random simulation index.

    Parameters:
    -----------
    data_instance : object
        The instance of the data containing the results and necessary attributes for indexing.
    data_type : str, optional
        The type of data to be indexed. Can be `"X"` for state results or `"Z"` for latent results. Defaults to `"X"`.
    num_plots_max : int, optional
        The maximum number of plots to be generated. Defaults to 6.
    idx_gen : str, optional
        Method for generating indices. Options are `"rand"` for random indices or `"first"` for sequential indices. Defaults to `"rand"`.

    Returns:
    -----------
    idx_n_n : np.ndarray
        Indices for the state variables (nodes).
    idx_n_dn : np.ndarray
        Indices for the state variables (degrees of freedom).
    idx_sim : int
        Random index for simulation selection.
    idx_n_f : np.ndarray
        Indices for the feature variables (states or latent variables).
    num_plots : int
        The actual number of plots to be generated.
    """
    # initialize return values
    idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots = [None] * 5

    if data_type == "X" or data_type == "x":
        # state results
        n_f = data_instance.n_n * data_instance.n_dn
        if n_f > num_plots_max:
            num_plots = num_plots_max
            match idx_gen:
                case "rand":
                    idx_n_n = np.random.randint(
                        0, data_instance.n_n, size=(num_plots_max,)
                    )
                    idx_n_dn = np.random.randint(
                        0, data_instance.n_dn, size=(num_plots_max,)
                    )
                case "first":
                    if data_instance.n_n < num_plots_max:
                        logging.info(
                            f"number of nodes {data_instance.n_n} smaller than maximum number of plots {num_plots_max}. Choosing random indices for the nodes."
                        )
                        idx_n_n = np.random.randint(
                            0, data_instance.n_n, size=(num_plots_max,)
                        )
                    else:
                        idx_n_n = np.arange(num_plots_max)
                    if data_instance.n_dn < num_plots_max:
                        logging.info(
                            f"number of node DOFs {data_instance.n_dn} smaller than maximum number of plots {num_plots_max}. Choosing random indices for the node DOFs."
                        )
                        idx_n_dn = np.random.randint(
                            0, data_instance.n_dn, size=(num_plots_max,)
                        )
                    else:
                        idx_n_dn = np.arange(num_plots_max)
                case _:
                    raise ValueError(f"Wrong idx_gen type {idx_gen}")
        else:
            num_plots = n_f
            # use all dofs
            idx_n_n = np.arange(data_instance.n_n)
            idx_n_dn = np.arange(data_instance.n_dn)

        # feature idx
        idx_n_f = []
        for i in range(num_plots):
            idx_n_f.append(
                np.random.choice(idx_n_n) * data_instance.n_dn
                + np.random.choice(idx_n_dn)
            )
        idx_n_f = np.array(idx_n_f)

    elif data_type == "Z" or data_type == "z":
        # latent results
        n_f = data_instance.n_red
        if n_f > num_plots_max:
            num_plots = num_plots_max
            match idx_gen:
                case "rand":
                    rng = np.random.default_rng()  # without repitition
                    idx_n_f = rng.choice(n_f, size=(num_plots_max,), replace=False)
                case "first":
                    idx_n_f = np.arange(num_plots_max)
                case _:
                    raise ValueError(f"Wrong idx_gen type {idx_gen}")
        else:
            num_plots = n_f
            idx_n_f = np.arange(n_f)
    else:
        raise ValueError(f"Data type {data_type} unknown.")

    # random simulation
    idx_sim = np.random.randint(0, data_instance.n_sim)

    return idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots


def plot_x(num_plots, x, x_id, idx_n_f, variable_names, save_name=None, save_path=""):
    """
    Plots and compares multiple time series data from the state in the feature shape.
    Data needs to be in the feature format (n_s, n_f)

    This function creates a series of subplots to compare the provided time series data (`x`) with identified data (`x_id`)
    for multiple features. Each subplot shows the data for one feature, with the original data plotted as a solid line and
    the identified data plotted as a dashed line. The function allows for saving the resulting figure as a PNG file.

    Parameters:
    -----------
    num_plots : int
        The number of subplots to create. Determines how many features will be plotted.

    x : np.ndarray of size (n_s, n_f)
        Array containing the original time series data. Expected shape is (n_time_steps, n_features).

    x_id : np.ndarray of size (n_s, n_f)
        Array containing the identified or reconstructed time series data. Expected shape is (n_time_steps, n_features).

    idx_n_f : list or np.ndarray
        Indices of the features to be plotted. Should have the same length as `num_plots`.

    variable_names : list of str
        Names of the variables to be used in the plot labels. Should contain two elements: the name for `x` and the name for `x_id`.

    save_name : str, optional
        The name of the file to save the plot as (without file extension). If None, uses the first element of `variable_names`.

    save_path : str, optional
        Directory where the plot will be saved. If not provided, the plot will be saved in the current working directory.

    Returns:
    --------
    None
    """
    if save_name == None:
        save_name = variable_names[0]

    fig, ax = new_fig(num_plots)
    for i, i_n_f in enumerate(idx_n_f):
        ax[i].plot(x[:, i_n_f], label=rf"${variable_names[0]}$")
        ax[i].plot(x_id[:, i_n_f], linestyle="dashed", label=rf"${variable_names[1]}$")
        ax[i].set_ylabel(
            rf"$f_{{{i_n_f + 1}}}$",
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax[i].grid(linestyle=":", linewidth=1)
    plt.xlabel("time t [s]")
    fig.align_ylabels(ax[:])
    plt.legend(bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0.0)
    # fig.legend(loc='outside center right', bbox_to_anchor=(1.3, 0.6))
    plt.tight_layout()
    plt.show(block=False)
    save_as_png(os.path.join(save_path, save_name))


def plot_X(
    num_plots,
    t,
    X,
    X_id,
    idx_n_n,
    idx_n_dn,
    idx_sim,
    variable_names,
    save_name=None,
    save_path="",
):
    """
    Plots multiple time series of the states from the provided data and optionally saves the plot.
    Data needs to be in the state format (n_sim, n_t, n_n, n_dn).

    This function creates a series of plots comparing the time evolution of system variables
    across different indices. Each plot shows the simulated variable and its corresponding reference
    or ideal value. The function also supports saving the plot to a specified directory.

    Parameters:
    -----------
    num_plots : int
        The number of individual plots to generate.
    t : numpy.ndarray
        The time vector for the x-axis.
    X : numpy.ndarray of size (n_sim, n_t, n_n, n_dn)
        The simulated data array with shape (num_simulations, num_time_points, num_n, num_dn).
    X_id : numpy.ndarray of size (n_sim, n_t, n_n, n_dn)
        The reference or ideal data array with the same shape as `X`.
    idx_n_n : list of int
        List of indices for the 'n_n' dimension of `X` and `X_id`.
    idx_n_dn : list of int
        List of indices for the 'n_dn' dimension of `X` and `X_id`.
    idx_sim : int
        Index of the simulation to plot.
    variable_names : list of str
        List of two variable names used for labeling the plot.
    save_name : str, optional
        The name of the file to save the plot as. If `None`, uses the first variable name from `variable_names`.
    save_path : str, optional
        The directory path where the plot image will be saved if `save_name` is provided. Defaults to the current directory.
    """

    if save_name == None:
        save_name = variable_names[0]

    fig, ax = new_fig(num_plots)
    # get random n_n, n_dn combinations
    indices = random.sample(
        sorted(itertools.product(idx_n_n.tolist(), idx_n_dn.tolist())), num_plots
    )
    for i, index in enumerate(indices):
        i_n = index[0]
        i_dn = index[1]
        ax[i].plot(t, X[idx_sim, :, i_n, i_dn], label=rf"${variable_names[0]}$")
        ax[i].plot(
            t,
            X_id[idx_sim, :, i_n, i_dn],
            linestyle="dashed",
            label=rf"${variable_names[1]}$",
        )
        ax[i].set_ylabel(
            rf"$n_{{{i_n + 1}}}/dn_{{{i_dn + 1}}}$",
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax[i].grid(linestyle=":", linewidth=1)
    plt.xlabel("time t [s]")
    fig.align_ylabels(ax[:])
    plt.legend(bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0.0)
    # fig.legend(loc='outside center right', bbox_to_anchor=(1.3, 0.6))
    plt.tight_layout()
    plt.show(block=False)
    save_as_png(os.path.join(save_path, save_name))


def plot_Z(
    num_plots,
    t,
    Z,
    Z_id,
    idx_n_f,
    idx_sim,
    variable_names,
    save_name=None,
    save_path="",
):
    """
    Plots the comparison between original and identified latent variables over time.
    Data needs to be in the latent state format (n_sim, n_t, n_f).

    This function generates a series of subplots comparing the original latent variables (`Z`) with their identified counterparts
    (`Z_id`) for a specified number of features. The plots are generated for a selected simulation, and each subplot represents a
    different feature.

    Parameters:
    -----------
    num_plots : int
        The number of subplots to generate.

    t : array-like
        The time variable, common to both the original and identified data.

    Z : array-like of size (n_sim, n_t, n_f).
        The original latent variables, with shape (num_simulations, num_time_steps, num_features).

    Z_id : array-like of size (n_sim, n_t, n_f).
        The identified or reconstructed latent variables, with the same shape as `Z`.

    idx_n_f : array-like
        Indices of the selected features to plot.

    idx_sim : int
        Index of the simulation to plot.

    variable_names : list of str
        A list containing the names of the variables to be used in the plot labels. The first element should be the label for `Z`,
        and the second element should be the label for `Z_id`.

    save_name : str, optional
        The base name for the saved plot image. If not provided, the first element of `variable_names` is used. Default is None.

    save_path : str, optional
        The directory path where the plot image will be saved. Default is an empty string (current directory).

    Returns:
    --------
    None
        The function generates and displays the plots, and saves the image to the specified path.
    """
    if save_name == None:
        save_name = variable_names[0]

    fig, ax = new_fig(num_plots)
    i = 0
    for i, i_n_f in enumerate(idx_n_f):
        ax[i].plot(t, Z[idx_sim, :, i_n_f], label=rf"${variable_names[0]}$")
        ax[i].plot(
            t,
            Z_id[idx_sim, :, i_n_f],
            linestyle="dashed",
            label=rf"${variable_names[1]}$",
        )
        ax[i].set_ylabel(
            rf"$f_{{{i_n_f+1}}}$",
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax[i].grid(linestyle=":", linewidth=1)
        i += 1
    plt.xlabel("time t [s]")
    fig.align_ylabels(ax[:])
    plt.legend(bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0.0)
    # fig.legend(loc='outside center right', bbox_to_anchor=(1.3, 0.6))
    plt.tight_layout()
    plt.show(block=False)
    save_as_png(os.path.join(save_path, save_name))


def plot_z(num_plots, z, z_id, idx_n_f, variable_names, save_name=None, save_path=""):
    """
    Plots a comparison between original and identified latent variables.
    Data needs to be in the feature format (n_s, n_f).

    This function creates a series of subplots to compare the original latent variables (`z`) with the identified or reconstructed
    ones (`z_id`). Each subplot corresponds to a specific feature as indicated by `idx_n_f`.

    Parameters:
    -----------
    num_plots : int
        The number of subplots to create, corresponding to the number of features to plot.

    z : array-like, shape (n_s, n_f)
        The original latent variables, where `n_s` is the number of samples (time steps) and `n_f` is the number of features.

    z_id : array-like, shape (n_s, n_f)
        The identified or reconstructed latent variables, with the same shape as `z`.

    idx_n_f : array-like
        Indices of the selected features to plot, corresponding to columns of `z` and `z_id`.

    variable_names : list of str
        A list containing the names of the variables for the plot labels. The first element should correspond to `z`, and the
        second element to `z_id`.

    save_name : str, optional
        The base name for the saved plot image. If not provided, the first element of `variable_names` is used. Default is None.

    save_path : str, optional
        The directory path where the plot image will be saved. Default is an empty string (current directory).

    Returns:
    --------
    None
        The function generates and displays the plots, and saves the image to the specified path.
    """
    if save_name == None:
        save_name = variable_names[0]

    fig, ax = new_fig(num_plots)
    i = 0
    for i, i_nf in enumerate(idx_n_f):
        ax[i].plot(z[:, i_nf], label=rf"${variable_names[0]}$")
        ax[i].plot(z_id[:, i_nf], linestyle="dashed", label=rf"${variable_names[1]}$")
        ax[i].set_ylabel(
            rf"$f_{{{i_nf+1}}}$",
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax[i].grid(linestyle=":", linewidth=1)
        i += 1
    plt.xlabel("time t [s]")
    fig.align_ylabels(ax[:])
    plt.legend(bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0.0)
    # fig.legend(loc='outside center right', bbox_to_anchor=(1.3, 0.6))
    plt.tight_layout()
    save_as_png(os.path.join(save_path, save_name))
    plt.show(block=False)


def new_fig(num_plots):
    """
    Creates a new figure with subplots for plotting.

    This function generates a figure and a set of subplots that can be used for plotting multiple variables or features
    in a vertically stacked layout.

    Parameters:
    -----------
    num_plots : int
        The number of subplots to create. Each subplot will be arranged vertically in a single column.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object containing the subplots.

    ax : numpy.ndarray or matplotlib.axes.Axes
        An array of Axes objects for the subplots. If `num_plots` is 1, `ax` will be a single Axes object;
        otherwise, it will be an array of Axes objects.
    """
    fig, ax = plt.subplots(num_plots, 1, figsize=(5.701, 3.5), dpi=300, sharex="all")
    return fig, ax


def save_as_png(save_path):
    """
    Saves the current matplotlib figure as a PNG file.

    This function saves the current matplotlib figure to the specified path as a PNG file. If the provided path does not
    end with ".png", the function appends ".png" to the path. If a file already exists at the path, it is removed before
    saving the new PNG file. The function handles potential runtime errors related to saving figures, including issues with
    LaTeX processing and unsupported backends. If an error occurs, the function attempts to use different matplotlib backends
    to save the figure.

    Parameters:
    -----------
    save_path : str
        The file path where the PNG file will be saved. The path should include the desired filename and, optionally, the directory.

    Returns:
    -----------
    None
    """
    if save_path.endswith(".png"):
        pass
    else:
        save_path = f"{save_path}.png"
    if os.path.isfile(save_path):
        # remove old .png to prevent confusion with old results
        os.remove(save_path)

    # save new .png
    try:
        plt.savefig(save_path)
    except RuntimeError:
        # sometimes "Failed to process string with tex because dvipng could not be found" error occurs
        try:
            import matplotlib as mpl

            mpl.rcParams.update(mpl.rcParamsDefault)  # default mpl helps sometimes
            plt.savefig(save_path)
            # if it still occurs change the backend
        except RuntimeError:
            # change backend
            # see https://matplotlib.org/stable/users/explain/figure/backends.html for list of backends

            setup_matplotlib()  # reset to tex
            backends = [
                "qtagg",
                "ipympl",
                "tkagg",
                "macosx",
                "pdf",
            ]
            for backend in backends:
                try:
                    mpl.use(backend)
                    if backend == "pdf":
                        save_path_tmp = f"{os.path.splitext(save_path)[0]}.pdf"
                    else:
                        save_path_tmp = save_path
                    plt.savefig(save_path_tmp)
                    return
                except:
                    # go to next backend
                    pass
            logging.error(f"Plot {save_path} could not be created due to RunTimeError")


def plot_time_trajectories_all(
    data, data_id, use_train_data=False, idx_gen="rand", result_dir=""
):
    """
    Generates and saves a series of plots comparing and reconstructing time trajectories of state and latent features.

    This function iterates through predefined lists of plotting functions for state and latent features, calling each
    function to generate and save the corresponding plots. It allows for plotting time trajectories, comparing
    reconstructed states, and visualizing latent feature errors and reconstructions.

    Parameters:
    -----------
    data : object
        The dataset containing state and latent feature information. This object should have attributes suitable for
        plotting functions such as `plot_X_comparison` and `plot_Z_ph`.

    data_id : object
        The dataset containing identified features and results used for comparison and reconstruction. This object should
        have attributes for plotting functions that need identified data.

    use_train_data : bool, optional
        If True, the function will use the training data for plotting. If False, it will use the test data. Default is False.

    idx_gen : str, optional
        Method for generating indices of features to plot. Options are:
        - "rand": Randomly selects indices.
        - "first": Selects the first few indices.
        Default is "rand".

    result_dir : str, optional
        The directory path where the plots will be saved. If not provided, the plots will not be saved. Default is an empty string.

    Returns:
    --------
    None
        The function does not return anything. It generates and displays plots, and optionally saves them to files.
    """
    state_functions = [
        plot_X_comparison,
        plot_X_dt_comparison,
        plot_x_comparison,
        plot_x_reconstruction,
        plot_x_dt_reconstruction,
        plot_X_reconstruction,
        plot_X_dt_reconstruction,
    ]
    latent_state_functions = [
        plot_Z_ph,
        plot_z_ph,
        plot_Z_dt_ph_map,
        plot_z_dt_ph_map,
        plot_Z_dt_ph,
        plot_z_dt_ph,
    ]

    for state_function in state_functions:
        state_function(
            data,
            data_id,
            use_train_data=use_train_data,
            idx_gen=idx_gen,
            save_path=result_dir,
        )

    for latent_state_function in latent_state_functions:
        latent_state_function(
            data_id,
            use_train_data=use_train_data,
            idx_gen=idx_gen,
            save_path=result_dir,
        )


def chessboard_visualisation(test_ids, system_layer, data, result_dir):
    """
    Visualizes and compares predicted and test matrices by generating various plots.

    This function generates a comprehensive visualization of the predicted and test matrices from a system layer.
    It creates a grid of images showing the matrices and their absolute errors, saves these images, and exports
    the colormap used for visualizations. Additionally, it visualizes the matrices as animations and saves them as images.

    Parameters:
    -----------
    test_ids : list of int
        List of indices for the test samples to visualize and compare.

    system_layer : object
        The system layer object that provides methods to get predicted system matrices.

    data : object
        The dataset containing test data and matrices. This object should have attributes such as `test_data` and `ph_matrices_test`.

    result_dir : str
        The directory path where the result images, colormaps, and limits will be saved.

    Returns:
    --------
    None
        The function generates and saves various plots and files related to matrix visualizations and comparisons.
    """
    mu_test = data.test_data[-1]
    n_sim_test, n_t_test, _, _, _, _ = data.shape_test
    # predicted matrices
    J_pred, R_pred, B_pred = system_layer.get_system_matrices(mu_test, n_t=n_t_test)
    # original test matrices
    J_test_, R_test_, Q_test_, B_test_ = data.ph_matrices_test

    J_pred, R_pred, B_pred = (
        J_pred[test_ids],
        R_pred[test_ids],
        B_pred[test_ids],
    )
    J_test_, R_test_, B_test_ = (
        J_test_[test_ids],
        R_test_[test_ids],
        B_test_[test_ids],
    )

    J_min, J_max = min(J_pred.min(), J_test_.min()), max(J_pred.max(), J_test_.max())
    R_min, R_max = min(R_pred.min(), R_test_.min()), max(R_pred.max(), R_test_.max())
    B_min, B_max = min(B_pred.min(), B_test_.min()), max(B_pred.max(), B_test_.max())

    e_J = np.abs(J_pred - J_test_)  # / J_test[test_ids].max()
    e_R = np.abs(R_pred - R_test_)  # / R_test_.max()
    e_B = np.abs(B_pred - B_test_)  # / B_test[test_ids].max()

    np.linalg.matrix_rank(B_test_[0])

    fig, axs = plt.subplots(9, max(len(test_ids), 2))
    color_factor = 1
    for i, test_id in enumerate(test_ids):
        axs[0, i].imshow(
            J_pred[i], vmin=color_factor * J_min, vmax=color_factor * J_max
        )
        axs[1, i].imshow(
            J_test_[i], vmin=color_factor * J_min, vmax=color_factor * J_max
        )
        axs[2, i].imshow(e_J[i], vmin=color_factor * J_min, vmax=color_factor * J_max)
        axs[3, i].imshow(
            R_pred[i], vmin=color_factor * R_min, vmax=color_factor * R_max
        )
        axs[4, i].imshow(
            R_test_[i], vmin=color_factor * R_min, vmax=color_factor * R_max
        )
        axs[5, i].imshow(e_R[i], vmin=color_factor * R_min, vmax=color_factor * R_max)
        axs[6, i].imshow(
            B_pred[i], vmin=color_factor * B_min, vmax=color_factor * B_max
        )
        axs[7, i].imshow(
            B_test_[i], vmin=color_factor * B_min, vmax=color_factor * B_max
        )
        axs[8, i].imshow(e_B[i], vmin=color_factor * B_min, vmax=color_factor * B_max)
        if i == 0:
            axs[0, i].set_ylabel("J_pred")
            axs[1, i].set_ylabel("J_test")
            axs[2, i].set_ylabel("e_J")
            axs[3, i].set_ylabel("R_pred")
            axs[4, i].set_ylabel("R_test")
            axs[5, i].set_ylabel("e_R")
            axs[6, i].set_ylabel("B_pred")
            axs[7, i].set_ylabel("B_test")
            axs[8, i].set_ylabel("e_B")

    plt.savefig(
        os.path.join(result_dir, f"compare_matrices.png"),
        bbox_inches="tight",
        pad_inches=0,
    )

    # %% save matrices as figures
    plt.figure()
    t_test, x_test, dx_dt_test, u_test, mu_test = data.test_data
    for u_ in u_test:
        plt.plot(u_)
    cmap = "plasma"
    i = 0
    for i in range(5):
        for key, matrix in dict(
            J=dict(
                limits=[J_min, J_max],
                matrices=dict(pred=J_pred, ref=J_test_, error=e_J),
            ),
            R=dict(
                limits=[R_min, R_max],
                matrices=dict(pred=R_pred, ref=R_test_, error=e_R),
            ),
            B=dict(
                limits=[B_min, B_max],
                matrices=dict(pred=B_pred, ref=B_test_, error=e_B),
            ),
        ).items():
            for sub_key, sub_matrix in matrix["matrices"].items():
                # plot without gui and save figure
                plt.figure()
                plt.imshow(
                    sub_matrix[i],
                    vmin=matrix["limits"][0],
                    vmax=matrix["limits"][1],
                    cmap=cmap,
                )
                # remove ticks
                plt.xticks([])
                plt.yticks([])
                # plt.show()
                # save figure
                plt.savefig(
                    os.path.join(result_dir, f"{key}_{sub_key}_{i}.png"),
                    bbox_inches="tight",
                    pad_inches=0,
                )
    # save txt file with upper and lower limits
    with open(os.path.join(result_dir, "limits.txt"), "w") as file:
        file.write(f"J: min: {J_min} max: {J_max}\n")
        file.write(f"R: min: {R_min} max: {R_max}\n")
        file.write(f"B: min: {B_min} max: {B_max}\n")

    # export colormap to tikz
    def convert_rgb(l):
        return [int(l[0] * 255), int(l[1] * 255), int(l[2] * 255)]

    def colormap_to_tikz(colormap, name):
        output = "{" + name + "}{\n"

        for i in range(colormap.N):
            rgb = colormap(i)[:3]
            newcolor = convert_rgb(rgb)
            output += f"    rgb255=({newcolor[0]},{newcolor[1]},{newcolor[2]});\n"
        output += "}\n"
        return output

    colormap = plt.get_cmap(cmap)

    result = open(f"{cmap}.txt", "w")
    result.write(colormap_to_tikz(colormap, "gnuplot2"))
    result.close()

    # %% visualize
    # plot comparison of original and reconstructed data as animation
    min_values = [J_pred.min(), R_pred.min(), B_pred.min()]
    max_values = [J_pred.max(), R_pred.max(), B_pred.max()]

    fig, ax = plt.subplots(1, 4, figsize=(12, 3), dpi=300, sharex="all", sharey="all")
    # global title
    # imshow matrices
    im1 = ax[0].imshow(J_pred[0], vmin=min_values[0], vmax=max_values[0])
    ax[0].set_title("$J_{ph}$")
    im2 = ax[1].imshow(R_pred[0], vmin=min_values[1], vmax=max_values[1])
    ax[1].set_title("$R_{ph}$")
    im3 = ax[2].imshow(B_pred[0], vmin=min_values[2], vmax=max_values[2])
    ax[2].set_title("$B_{ph}$")

    fig.tight_layout()


def plot_train_history(train_hist):
    """
    Plots the training history of a machine learning model.

    This function generates a plot of the training history for different loss metrics,
    including the overall loss, derivative losses, reconstruction loss, and regularization loss.
    The losses are plotted on a logarithmic scale to better visualize the convergence and performance
    of the training process.

    Parameters:
    -----------
    train_hist : object
        An object containing the training history. This object should have a `history` attribute,
        which is a dictionary with keys corresponding to different loss metrics (e.g., "loss", "dz_loss",
        "dx_loss", "rec_loss", "reg_loss") and values as lists or arrays of loss values recorded during training.

    Returns:
    --------
    None
        The function does not return any value. It generates and displays a plot of the training history.
    """
    # plot training history
    plt.figure()
    plt.semilogy(train_hist.history["loss"], label="loss")
    plt.semilogy(train_hist.history["dz_loss"], label="dz")
    try:
        plt.semilogy(train_hist.history["dx_loss"], label="dx")
        plt.semilogy(train_hist.history["rec_loss"], label="rec")
    except KeyError:
        pass
    plt.semilogy(train_hist.history["reg_loss"], label="reg")
    plt.legend()
