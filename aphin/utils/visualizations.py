"""
Utilities for the use in the examples
"""

import os
import re
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from aphin.layers import PHQLayer, PHLayer
from PIL import Image
from natsort import natsorted

import pandas as pd
import itertools

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
        # Ensure the "results" directory exists
        os.makedirs("results", exist_ok=True)
        # Use PGF backend for saving plots
        matplotlib.use("pgf")
    else:
        matplotlib.use("TkAgg")  # Interactive backend for display

    # Define common LaTeX preamble
    latex_preamble = "\n".join(
        [
            r"\usepackage{amsmath}",
            r"\usepackage{bm}",
        ]
    )

    # Update matplotlib rcParams (only once)
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",  # Choose either "pdflatex" or "lualatex"
            "pgf.rcfonts": False,
            "text.usetex": True,
            "text.latex.preamble": latex_preamble,
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
    original_data,
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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
        original_data,
        identified_data,
        "X",
        "X",
        use_train_data,
        "X",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
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
    original_data,
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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
        original_data,
        identified_data,
        "x",
        "x",
        use_train_data,
        "x",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
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
    original_data,
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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
            original_data,
            identified_data,
            "X_dt",
            "X_dt",
            use_train_data,
            "X",
            idx_gen,
            idx_custom_tuple=idx_custom_tuple,
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
    original_data,
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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
        original_data,
        identified_data,
        "x",
        "x_rec",
        use_train_data,
        "x",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
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
    original_data,
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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
        idx_custom_tuple=idx_custom_tuple,
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
    original_data,
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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
        original_data,
        identified_data,
        "X",
        "X_rec",
        use_train_data,
        "X",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
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
    original_data,
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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
        idx_custom_tuple=idx_custom_tuple,
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


def plot_Z_ph(
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
):
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
        identified_data,
        "Z",
        "Z_ph",
        use_train_data,
        "Z",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
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


def plot_z_ph(
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
):
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
        identified_data,
        "z",
        "z_ph",
        use_train_data,
        "z",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
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
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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

    t, Z_dt, Z_dt_ph_map, idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots = (
        get_quantities_of_interest(
            identified_data,
            "Z_dt",
            "Z_dt_ph_map",
            use_train_data,
            "Z",
            idx_gen,
            idx_custom_tuple=idx_custom_tuple,
        )
    )

    if Z_dt_ph_map is not None:
        variable_names = [r"\dot{\bm{Z}}", r"\dot{\bm{Z}}_{\mathrm{phmap}}"]
        save_name = "Z_dt_ph_map"
        plot_Z(
            num_plots,
            t,
            Z_dt,
            Z_dt_ph_map,
            idx_n_f,
            idx_sim,
            variable_names,
            save_name=save_name,
            save_path=save_path,
        )


def plot_z_dt_ph_map(
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
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

    t, z_dt, z_dt_ph_map, _, _, _, idx_n_f, num_plots = get_quantities_of_interest(
        identified_data,
        "z_dt",
        "z_dt_ph_map",
        use_train_data,
        "z",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
    )

    if z_dt_ph_map is not None:
        variable_name = [r"\dot{\bm{z}}", r"\dot{\bm{z}}_{\mathrm{phmap}}"]
        save_name = "z_dt_ph_map"
        plot_z(
            num_plots,
            z_dt,
            z_dt_ph_map,
            idx_n_f,
            variable_name,
            save_name=save_name,
            save_path=save_path,
        )


def plot_Z_dt_ph(
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
):
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
        identified_data,
        "Z_dt",
        "Z_dt_ph",
        use_train_data,
        "Z",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
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


def plot_z_dt_ph(
    identified_data,
    use_train_data=False,
    idx_gen="rand",
    save_path="",
    idx_custom_tuple: list[tuple] | None = None,
):
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
        identified_data,
        "z_dt",
        "z_dt_ph",
        use_train_data,
        "z",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
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
    id_quantity_1: str,
    id_quantity_2: str,
    use_train_data=False,
    data_type="X",
    idx_gen="rand",
    idx_custom_tuple: list[tuple] | None = None,
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
        data,
        data_type=data_type,
        idx_gen=idx_gen,
        idx_custom_tuple=idx_custom_tuple,
    )

    t = data.t
    quantity_1 = getattr(data, id_quantity_1)
    quantity_2 = getattr(data, id_quantity_2)

    for quantity, quantity_name in [
        (quantity_1, id_quantity_1),
        (quantity_2, id_quantity_2),
    ]:
        if quantity is None:
            logging.info(
                f"Quantity {quantity_name} is not given in the dataset. Plot will not be created."
            )

    return t, quantity_1, quantity_2, idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots


def get_quantity_of_interest(
    original_data,
    identified_data,
    og_quantity: str,
    id_quantity: str,
    use_train_data=False,
    data_type="X",
    idx_gen="rand",
    idx_custom_tuple: list[tuple] | None = None,
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
        original_data,
        data_type=data_type,
        idx_gen=idx_gen,
        idx_custom_tuple=idx_custom_tuple,
    )

    t = identified_data.t
    quantity_1 = getattr(original_data, og_quantity)
    quantity_2 = getattr(identified_data, id_quantity)

    for quantity, quantity_name in [
        (quantity_1, og_quantity),
        (quantity_2, id_quantity),
    ]:
        if quantity is None:
            logging.info(
                f"Quantity {quantity_name} is not given in the dataset. Plot will not be created."
            )

    return t, quantity_1, quantity_2, idx_n_n, idx_n_dn, idx_sim, idx_n_f, num_plots


def plot_u(
    data,
    use_train_data=False,
    num_plots_max=6,
):
    if use_train_data:
        data = data.TRAIN
    else:
        data = data.TEST
    t = data.t
    num_plots = data.n_u
    u = data.u

    if num_plots > num_plots_max:
        num_plots = num_plots_max

    fig, ax = new_fig(num_plots, window_title="Input u")
    if num_plots == 1:
        ax = [ax]
    ax[0].set_title("Inputs")
    for i_u in range(num_plots):
        ax[i_u].plot(u[:, i_u], label=rf"$u_{i_u}$")
        ax[i_u].set_ylabel(
            rf"$u_{i_u}$",
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax[i_u].grid(linestyle=":", linewidth=1)
        ax[i_u].legend()
    # plt.xlabel("time t [s]")
    fig.align_ylabels(ax[:])
    # plt.legend(bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0.0)
    # fig.legend(loc='outside center right', bbox_to_anchor=(1.3, 0.6))
    plt.tight_layout()
    plt.show(block=False)
    # save_as_png(os.path.join(save_path, save_name))


def plot_errors(
    data,
    use_train_data=False,
    # title_label="",
    save_name="rms_error",
    domain_names=None,
    save_to_csv=False,
    yscale="linear",
    create_train_test_subfolder: bool = False,
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

    if create_train_test_subfolder:
        if use_train_data:
            result_dir = os.path.join(result_dir, "train")
        else:
            result_dir = os.path.join(result_dir, "test")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

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
            t=data.t,
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
            t=data.t,
            title_label=title_label,
            save_name=save_name_lat,
            save_to_csv=save_to_csv,
            yscale=yscale,
        )


def single_parameter_space_error_plot(
    norm_rms_error,
    Mu,
    Mu_input: np.ndarray = None,
    parameter_names: list[str] = None,
    save_name: str = None,
):
    if Mu_input is not None:
        Mu_with_input = np.concatenate((Mu, Mu_input), axis=1)
    else:
        Mu_with_input = Mu
    n_parameter_space = Mu_with_input.shape[1]
    if n_parameter_space > 4:
        raise NotImplementedError(
            f"Parameter space of size {n_parameter_space} is too large. Please reduce to a size of at most 4."
        )
    if parameter_names is not None:
        assert len(parameter_names) == n_parameter_space

    # prepare scatter data
    x_data = Mu_with_input[:, 0]
    y_data = Mu_with_input[:, 1]
    if n_parameter_space > 2:
        z_data = Mu_with_input[:, 2]
    if n_parameter_space > 3:
        colors = Mu_with_input[:, 3]

    # size of scatter dots is the error
    norm_rms_error_max_time = np.max(norm_rms_error, axis=1)
    error_sizes = norm_rms_error_max_time
    size_min = 10
    size_max = 50
    error_sizes_scaled = (
        (error_sizes - error_sizes.min(axis=0))
        / (error_sizes.max(axis=0) - error_sizes.min(axis=0))
    ) * (size_max - size_min) + size_min

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    if n_parameter_space == 2:
        ax.scatter(x_data, y_data, sizes=error_sizes_scaled)
    elif n_parameter_space == 3:
        ax.scatter(x_data, y_data, z_data, sizes=error_sizes_scaled)
    elif n_parameter_space == 4:
        im = ax.scatter(x_data, y_data, z_data, c=colors, sizes=error_sizes_scaled)
        cbar = fig.colorbar(im, ax=ax)

    if parameter_names is not None:
        for i_dim in range(n_parameter_space):
            if i_dim == 0:
                ax.set_xlabel(parameter_names[0])
            elif i_dim == 1:
                ax.set_ylabel(parameter_names[1])
            elif i_dim == 2:
                ax.set_zlabel(parameter_names[2])
            elif i_dim == 3:
                cbar.set_label(parameter_names[3])

    plt.title(f"Error = Circle size")
    plt.show(block=False)


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
        if t.ndim == 1:
            t = t[:, np.newaxis]
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


def get_sim_idx(
    data_instance,
    data_type="X",
    num_plots_max=6,
    idx_gen="rand",
    idx_custom_tuple: list[tuple] | None = None,
):
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
                    rng = np.random.default_rng()

                    if data_instance.n_n < num_plots_max:
                        replace_n_n = True
                    else:
                        replace_n_n = False
                    idx_n_n = rng.choice(
                        data_instance.n_n,
                        size=(num_plots_max,),
                        replace=replace_n_n,
                    )
                    if data_instance.n_dn < num_plots_max:
                        replace_n_dn = True
                    else:
                        replace_n_dn = False
                    idx_n_dn = rng.choice(
                        data_instance.n_dn,
                        size=(num_plots_max,),
                        replace=replace_n_dn,
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
                case "custom":
                    idx_n_n = []
                    idx_n_dn = []
                    for idx_tuple in idx_custom_tuple:
                        assert idx_tuple[0] <= data_instance.n_n - 1
                        assert idx_tuple[1] <= data_instance.n_dn - 1
                        idx_n_n.append(idx_tuple[0])
                        idx_n_dn.append(idx_tuple[1])
                    idx_n_n = np.array(idx_n_n)
                    idx_n_dn = np.array(idx_n_dn)
                    num_plots = len(idx_n_n)
                case _:
                    raise ValueError(f"Wrong idx_gen type {idx_gen}")
        else:
            num_plots = n_f
            # use all dofs
            idx_n_n = np.arange(data_instance.n_n)
            idx_n_dn = np.arange(data_instance.n_dn)

        # feature idx from node and node DOF idx
        idx_nf_combinations = list(
            itertools.product(np.unique(idx_n_n), np.unique(idx_n_dn))
        )
        if len(idx_nf_combinations) <= num_plots_max:
            num_plots = len(idx_nf_combinations)
            list_n_f_comb = np.array(idx_nf_combinations)
        else:
            rng = np.random.default_rng()
            list_n_f_comb = np.squeeze(
                rng.choice(idx_nf_combinations, size=(num_plots_max,), replace=False)
            )
        idx_n_n = list_n_f_comb[:, 0]
        idx_n_dn = list_n_f_comb[:, 1]
        idx_n_f = list(list_n_f_comb[:, 0] * data_instance.n_dn + list_n_f_comb[:, 1])

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
                case "custom":
                    idx_n_f = []
                    for idx_tuple in idx_custom_tuple:
                        assert idx_tuple[2] <= n_f - 1
                        idx_n_f.append(idx_tuple[2])
                    num_plots = len(idx_n_f)
                case _:
                    raise ValueError(f"Wrong idx_gen type {idx_gen}")
        else:
            num_plots = n_f
            idx_n_f = np.arange(n_f)
    else:
        raise ValueError(f"Data type {data_type} unknown.")

    # random simulation
    idx_sim = np.random.randint(0, data_instance.n_sim)
    # TODO: Remove next line!!!
    idx_sim = 0

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

    fig, ax = new_fig(num_plots, window_title=save_name)
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

    bar_plot_all_features = True
    if bar_plot_all_features:
        fig, ax = new_fig(1, window_title=save_name)
        # use norm
        normalization_value = np.linalg.norm(x, axis=0).clip(
            1e-10
        )  # set threshold to 1e-10
        rel_error_for_each_state = (
            np.linalg.norm(x - x_id, axis=0)
        ) / normalization_value
        domain_names = ["T", "disp_x", "disp_y", "disp_z", "vel_x", "vel_y", "vel_z"]
        if domain_names is not None:
            num_groups = rel_error_for_each_state.shape[0] // len(
                domain_names
            )  # should equal number of nodes
            colors = ["red", "blue", "green", "cyan", "purple", "black", "pink"]
            color_list = colors[: len(domain_names)] * num_groups
            label_list = domain_names
            label_list.extend(
                [f"_{domain_name}" for domain_name in domain_names] * (num_groups - 1)
            )
        else:
            color_list = None
            label_list = None
        # use max
        # rel_error_for_each_state = np.max(np.abs(x - x_id), axis=0) / np.max(
        #     np.abs(x), axis=0
        # )
        # np.savetxt(f"error_for_each_state_{save_name}", rel_error_for_each_state)
        plt.bar(
            np.arange(x.shape[1]) + 1,
            rel_error_for_each_state,
            log=True,
            color=color_list,
            label=label_list,
        )
        plt.xlabel("features")
        plt.legend()
        plt.title(f"{save_name}")
        plt.show(block=False)
        save_as_png(os.path.join(save_path, f"bar_error_{save_name}"))


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

    fig, ax = new_fig(num_plots, window_title=save_name)
    # get n_n, n_dn combinations
    assert len(idx_n_dn) == len(idx_n_n)
    indices = [(idx_n_n[i], idx_n_dn[i]) for i in range(num_plots)]
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

    fig, ax = new_fig(num_plots, window_title=save_name)
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

    fig, ax = new_fig(num_plots, window_title=save_name)
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


def new_fig(num_plots, window_title: str | None = None):
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
    if window_title is not None:
        # set window title instead of "Figure 0", "Figure 1",...
        if matplotlib.get_backend() == "TkAgg" or "pdf":
            fig.canvas.manager.set_window_title(f"{window_title}")
        else:
            # TODO: Needs testing for other backends
            fig.canvas.set_window_title(f"{window_title}")

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
    data,
    data_id,
    use_train_data=False,
    idx_gen="rand",
    result_dir="",
    idx_custom_tuple: list[tuple] | None = None,
    create_train_test_subfolder: bool = False,
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
    if create_train_test_subfolder:
        if use_train_data:
            result_dir = os.path.join(result_dir, "train")
        else:
            result_dir = os.path.join(result_dir, "test")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

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
            idx_custom_tuple=idx_custom_tuple,
        )

    for latent_state_function in latent_state_functions:
        latent_state_function(
            data_id,
            use_train_data=use_train_data,
            idx_gen=idx_gen,
            save_path=result_dir,
            idx_custom_tuple=idx_custom_tuple,
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


def plot_train_history(train_hist, save_path: str = "", validation=False):
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

    if validation:
        val_strings = ["", "val_"]
    else:
        val_strings = [""]
    for val_str in val_strings:
        # plot training history
        plt.figure()
        plt.semilogy(train_hist.history[f"{val_str}loss"], label=f"{val_str}loss")
        plt.semilogy(train_hist.history[f"{val_str}dz_loss"], label=f"{val_str}dz_loss")
        try:
            plt.semilogy(
                train_hist.history[f"{val_str}dx_loss"], label=f"{val_str}dx_loss"
            )
            plt.semilogy(
                train_hist.history[f"{val_str}rec_loss"], label=f"{val_str}rec_loss"
            )
        except KeyError:
            pass
        plt.semilogy(
            train_hist.history[f"{val_str}reg_loss"], label=f"{val_str}reg_loss"
        )
        plt.legend()
        plt.show(block=False)
        save_name = f"{val_str}train_hist"
        save_as_png(os.path.join(save_path, save_name))


def custom_state_plot(
    data,
    data_id,
    attributes: list[str],
    index_list: list[tuple],
    train_or_test: str = "test",
    cut_time_idx: int | None = None,
    subplot_idx: list[int] = None,
    subplot_title: list[str] = None,
    legend: list[str] = None,
    result_dir: str = "",
):
    """
    indices list of tuples with size (n_sim,n_n,n_dn)
    attributes list of str with [attribute of data, attribute of data_id], e.g. ["X","X_ph"]
    legend list of entries for orig system
    """
    # get data
    if train_or_test.lower() == "train":
        data_train_or_test = data.TRAIN
        data_id_train_or_test = data_id.TRAIN

    elif train_or_test.lower() == "test":
        data_train_or_test = data.TEST
        data_id_train_or_test = data_id.TEST
    # get attribute
    state = getattr(data_train_or_test, attributes[0])
    state_id = getattr(data_id_train_or_test, attributes[1])

    t = data_train_or_test.t
    if cut_time_idx is None:
        n_t = t.shape[0]
    else:
        t = t[:cut_time_idx]
        n_t = t.shape[0]

    plot_state = np.zeros((len(index_list), n_t))
    plot_state_id = np.zeros((len(index_list), n_t))
    for i_index, index in enumerate(index_list):
        n_sim = index[0]
        n_n = index[1]
        n_dn = index[2]
        plot_state[i_index] = state[n_sim, :n_t, n_n, n_dn]
        plot_state_id[i_index] = state_id[n_sim, :n_t, n_n, n_dn]

    num_subplots = max(subplot_idx) + 1
    if legend is None:
        legend = [[None]] * len(index_list)

    # plot data
    width = 6
    height = 6 / 1.61  # golden ratio
    fig, ax = plt.subplots(
        num_subplots, 1, figsize=(width, height), dpi=600, sharex="all"
    )
    for i_index in range(len(index_list)):
        if num_subplots == 1:
            ax.plot(t, plot_state[i_index], label=rf"{legend[i_index]}")
            ax.plot(t, plot_state_id[i_index], "--")
            ax.legend()
            ax.title.set_text(subplot_title)
        else:
            ax[subplot_idx[i_index]].plot(
                t, plot_state[i_index], label=rf"{legend[i_index]}"
            )
            ax[subplot_idx[i_index]].plot(t, plot_state_id[i_index], "--")
            ax[subplot_idx[i_index]].legend()
            ax[subplot_idx[i_index]].title.set_text(subplot_title[subplot_idx[i_index]])
    plt.xlabel("Time in s")
    plt.savefig(os.path.join(result_dir, "custom_state_plot.png"))

    print("debug in custom_state_plot")


def compare_x_and_x_dt(
    data, use_train_data=True, idx_gen="rand", idx_custom_tuple=None
):

    t, x, dx_dt, _, _, _, idx_n_f, num_plots = get_quantities_of_interest(
        data,
        "x",
        "dx_dt",
        use_train_data,
        "x",
        idx_gen,
        idx_custom_tuple=idx_custom_tuple,
    )

    idx_n_f.append(1842 * 4 + 3)

    num_plots = 2
    fig, ax = new_fig(num_plots, window_title="")
    for i in range(num_plots):
        if i == 0:
            x_or_x_dt = x
            title = "x"
        else:
            x_or_x_dt = dx_dt
            title = "x_dt"
        ax[i].plot(x_or_x_dt[:, idx_n_f])
        ax[i].grid(linestyle=":", linewidth=1)
        ax[i].set_title(title)
        ax[i].legend([f"x{i_n_f}" for i_n_f in idx_n_f])
    plt.xlabel(f"Time")
    plt.show(block=False)
    # save_as_png(os.path.join(result_dir, save_name_coeff))


def get_weight_files(
    phin_or_aphin,
    weight_dir,
    weight_name_pre_weights,
    every_n_epoch=None,
    weight_indices=None,
):
    """
    Typically requires the fit to run with `save_many_weights=True` callback.
    """
    if every_n_epoch is not None and weight_indices is not None:
        raise ValueError(f"Specifiy either `every_n_epoch` or `weight_indices`")

    if not isinstance(phin_or_aphin.system_layer, PHLayer):
        raise TypeError(
            f"The system layer of ApHIN/pHIN must be a PHLayer. PHQLayer is not implemented."
        )

    # use tex for plots
    plt.rcParams["text.usetex"] = True

    # Lists to store the extracted numbers and matching files
    epoch_numbers = []
    weight_files = []

    # Regular expression to match the weight name pattern
    pattern = re.compile(rf"^{weight_name_pre_weights}(\d+)\.weights\.h5$")

    # Walk through the directory
    for root, dirs, files in os.walk(weight_dir):
        for file in files:
            match = pattern.match(file)
            if match:
                # Extract the number and convert it to an integer
                epoch_numbers.append(int(match.group(1)))
                # Store the full path of the matching file
                weight_files.append(os.path.join(root, file))

    if weight_files:
        logging.info(
            f"{len(weight_files)} files with a starting string `{weight_name_pre_weights}` were found."
        )
    else:
        raise ValueError(
            f"No weight files have been found with the starting string {weight_name_pre_weights} in directory {weight_dir}."
        )
    # sort alphabettically
    weight_files = np.array(natsorted(weight_files))
    epoch_numbers = np.array(natsorted(epoch_numbers))

    # choose indices
    if every_n_epoch is None and weight_indices is None:
        every_n_epoch = 5
        weight_indices = slice(0, len(weight_files), every_n_epoch)
    elif every_n_epoch:
        weight_indices = slice(0, len(weight_files), every_n_epoch)
    elif weight_indices is not None:
        weight_indices = weight_indices

    return weight_files, epoch_numbers, weight_indices


def plot_weight_coefficient_evolution(
    phin_or_aphin,
    data,
    result_dir,
    weight_dir,
    weight_name_pre_weights,
    every_n_epoch=None,
    weight_indices=None,
    sim_idx=0,
    save_name="coefficient_evolution",
    use_train_data=False,
):

    weight_files, epoch_numbers, weight_indices = get_weight_files(
        phin_or_aphin,
        weight_dir,
        weight_name_pre_weights,
        every_n_epoch=every_n_epoch,
        weight_indices=weight_indices,
    )

    if use_train_data:
        mu = data.data[-1]
        n_sim = data.TRAIN.shape[0]
        n_t = data.TRAIN.shape[1]
        idx_prefix = "train"
    else:
        mu = data.test_data[-1]
        n_sim = data.TEST.shape[0]
        n_t = data.TEST.shape[1]
        idx_prefix = "test"

    for i_weight_file, weight_file in enumerate(weight_files[weight_indices]):
        # logging.info()
        phin_or_aphin.load_weights(weight_file)
        if mu is not None:
            dof_J, dof_R, dof_B, dof_Q, _ = (
                phin_or_aphin.system_layer.get_parameter_dependent_weights(mu)
            )
        else:
            raise NotImplementedError(f"Non-parametric version not yet implemented.")

        # convert from Tensor to (n_sim, n_coeff)
        dof_J = np.reshape(dof_J.numpy(), (n_sim, n_t, dof_J.numpy().shape[1]))[:, 0, :]
        dof_R = np.reshape(dof_R.numpy(), (n_sim, n_t, dof_R.numpy().shape[1]))[:, 0, :]
        dof_B = np.reshape(
            dof_B.numpy(), (n_sim, n_t, dof_B.numpy().shape[1] * dof_B.numpy().shape[2])
        )[:, 0, :]
        dof_Q = np.reshape(dof_Q.numpy(), (n_sim, n_t, dof_Q.numpy().shape[1]))[:, 0, :]
        if i_weight_file == 0:
            # initialize arrays
            num_epochs = len(weight_files[weight_indices])
            J_coefficients = np.zeros((num_epochs, dof_J.shape[1]))
            R_coefficients = np.zeros((num_epochs, dof_R.shape[1]))
            B_coefficients = np.zeros((num_epochs, dof_B.shape[1]))
            Q_coefficients = np.zeros((num_epochs, dof_Q.shape[1]))

        J_coefficients[i_weight_file, :] = dof_J[sim_idx]
        R_coefficients[i_weight_file, :] = dof_R[sim_idx]
        B_coefficients[i_weight_file, :] = dof_B[sim_idx]
        Q_coefficients[i_weight_file, :] = dof_Q[sim_idx]

    coefficient_names = ["J", "R", "B", "Q"]
    coefficient_values = [
        J_coefficients,
        R_coefficients,
        B_coefficients,
        Q_coefficients,
    ]
    # plot coefficients
    for coefficient_name, coefficient_value in zip(
        coefficient_names, coefficient_values
    ):
        save_name_coeff = f"{save_name}_{coefficient_name}"
        fig, ax = new_fig(1, window_title=save_name_coeff)
        ax.plot(epoch_numbers[weight_indices], coefficient_value)
        ax.grid(linestyle=":", linewidth=1)
        plt.title(
            f"Coefficient evolution of {coefficient_name}, {idx_prefix} sim: {sim_idx}"
        )
        plt.xlabel(f"Epochs")
        plt.show(block=False)
        save_as_png(os.path.join(result_dir, save_name_coeff))


def create_gif(
    phin_or_aphin,
    data,
    result_dir,
    weight_dir,
    weight_name_pre_weights,
    test_idx,
    every_n_epoch=None,
    weight_indices=None,
    duration=1,
    loop=0,
    dpi=600,
):
    assert isinstance(test_idx, int)

    weight_files, epoch_numbers, weight_indices = get_weight_files(
        phin_or_aphin,
        weight_dir,
        weight_name_pre_weights,
        every_n_epoch=every_n_epoch,
        weight_indices=weight_indices,
    )

    # over max and min values
    J_max_overall, R_max_overall, B_max_overall, e_max_overall = [-np.inf] * 4
    J_min_overall, R_min_overall, B_min_overall, e_min_overall = [np.inf] * 4

    # colormaps
    cmap_error = plt.get_cmap("inferno")

    for weight_file, epoch_number in zip(
        weight_files[weight_indices], epoch_numbers[weight_indices]
    ):
        phin_or_aphin.load_weights(weight_file)
        # (
        #     J_pred,
        #     R_pred,
        #     B_pred,
        #     J_test_,
        #     R_test_,
        #     B_test_,
        #     J_min,
        #     J_max,
        #     R_min,
        #     R_max,
        #     B_min,
        #     B_max,
        #     e_J,
        #     e_R,
        #     e_B,
        # ) = extract_matrices_and_errors(data, phin_or_aphin.system_layer, test_idx)

        # fig, axs = plt.subplots(3, 3)
        # color_factor = 1
        # fig.suptitle(f"Epoch {epoch_number}")
        # im = axs[0, 0].imshow(
        #     J_pred, vmin=color_factor * J_min, vmax=color_factor * J_max
        # )
        # im = axs[1, 0].imshow(
        #     J_test_, vmin=color_factor * J_min, vmax=color_factor * J_max
        # )
        # im = axs[2, 0].imshow(e_J, vmin=color_factor * J_min, vmax=color_factor * J_max)
        # im = axs[0, 1].imshow(
        #     R_pred, vmin=color_factor * R_min, vmax=color_factor * R_max
        # )
        # im = axs[1, 1].imshow(
        #     R_test_, vmin=color_factor * R_min, vmax=color_factor * R_max
        # )
        # im = axs[2, 1].imshow(e_R, vmin=color_factor * R_min, vmax=color_factor * R_max)
        # im = axs[0, 2].imshow(
        #     B_pred, vmin=color_factor * B_min, vmax=color_factor * B_max
        # )
        # im = axs[1, 2].imshow(
        #     B_test_, vmin=color_factor * B_min, vmax=color_factor * B_max
        # )
        # im = axs[2, 2].imshow(e_B, vmin=color_factor * B_min, vmax=color_factor * B_max)

        (
            J_pred,
            R_pred,
            B_pred,
            J_test_,
            R_test_,
            B_test_,
            _,
            _,
            _,
            _,
            _,
            _,
            e_J,
            e_R,
            e_B,
        ) = extract_matrices_and_errors(
            data, phin_or_aphin.system_layer, test_idx, returnJRBminmax=False
        )

        # normalize matrices
        normalize_min_value = -1
        normalize_max_value = 1
        J_test_, J_original_min, J_original_max = normalize_to_range(
            J_test_, min_val=normalize_min_value, max_val=normalize_max_value
        )
        R_test_, R_original_min, R_original_max = normalize_to_range(
            R_test_, min_val=normalize_min_value, max_val=normalize_max_value
        )
        B_test_, B_original_min, B_original_max = normalize_to_range(
            B_test_, min_val=normalize_min_value, max_val=normalize_max_value
        )

        J_pred = scale_array_with_orig_values(
            J_pred,
            J_original_min,
            J_original_max,
            min_val=normalize_min_value,
            max_val=normalize_max_value,
        )
        R_pred = scale_array_with_orig_values(
            R_pred,
            R_original_min,
            R_original_max,
            min_val=normalize_min_value,
            max_val=normalize_max_value,
        )
        B_pred = scale_array_with_orig_values(
            B_pred,
            B_original_min,
            B_original_max,
            min_val=normalize_min_value,
            max_val=normalize_max_value,
        )

        e_J = np.abs(J_pred - J_test_)  # / J_test[test_ids].max()
        e_R = np.abs(R_pred - R_test_)  # / R_test_.max()
        e_B = np.abs(B_pred - B_test_)  # / B_test[test_ids].max()

        J_max, J_min = J_pred.max(), J_pred.min()
        R_max, R_min = R_pred.max(), R_pred.min()
        B_max, B_min = B_pred.max(), B_pred.min()
        e_J_max, e_J_min = e_J.max(), e_J.min()
        e_R_max, e_R_min = R_pred.max(), e_R.min()
        e_B_max, e_B_min = e_B.max(), e_B.min()
        if J_max > J_max_overall:
            J_max_overall = J_max
        if R_max > R_max_overall:
            R_max_overall = R_max
        if B_max > B_max_overall:
            B_max_overall = B_max
        if J_min < J_min_overall:
            J_min_overall = J_min
        if R_min < R_min_overall:
            R_min_overall = R_min
        if B_min < B_min_overall:
            B_min_overall = B_min

        # error
        if min(e_J_min, e_R_min, e_B_min) < e_min_overall:
            e_min_overall = min(e_J_min, e_R_min, e_B_min)
        if max(e_J_max, e_R_max, e_B_max) > e_max_overall:
            e_max_overall = max(e_J_max, e_R_max, e_B_max)

        max_pred_overall = max(J_max_overall, R_max_overall, B_max_overall)
        min_pred_overall = max(J_min_overall, R_min_overall, B_min_overall)
        # overall_range = max_pred_overall - min_pred_overall
        # original_range = normalize_max_value - normalize_min_value
        test_max = max_pred_overall
        pred_max = max_pred_overall
        test_min = min_pred_overall
        pred_min = min_pred_overall
        # set to overall
        error_to_overall = False
        if error_to_overall:
            error_max = e_max_overall
            error_min = e_min_overall
        else:
            # set to fixed
            error_max = 0.5
            error_min = 0

    image_list = []
    for weight_file, epoch_number in zip(
        weight_files[weight_indices], epoch_numbers[weight_indices]
    ):
        phin_or_aphin.load_weights(weight_file)

        (
            J_pred,
            R_pred,
            B_pred,
            J_test_,
            R_test_,
            B_test_,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = extract_matrices_and_errors(
            data, phin_or_aphin.system_layer, test_idx, returnJRBminmax=False
        )

        # normalize matrices
        normalize_min_value = -1
        normalize_max_value = 1
        J_test_, J_original_min, J_original_max = normalize_to_range(
            J_test_, min_val=normalize_min_value, max_val=normalize_max_value
        )
        R_test_, R_original_min, R_original_max = normalize_to_range(
            R_test_, min_val=normalize_min_value, max_val=normalize_max_value
        )
        B_test_, B_original_min, B_original_max = normalize_to_range(
            B_test_, min_val=normalize_min_value, max_val=normalize_max_value
        )

        J_pred = scale_array_with_orig_values(
            J_pred,
            J_original_min,
            J_original_max,
            min_val=normalize_min_value,
            max_val=normalize_max_value,
        )
        R_pred = scale_array_with_orig_values(
            R_pred,
            R_original_min,
            R_original_max,
            min_val=normalize_min_value,
            max_val=normalize_max_value,
        )
        B_pred = scale_array_with_orig_values(
            B_pred,
            B_original_min,
            B_original_max,
            min_val=normalize_min_value,
            max_val=normalize_max_value,
        )

        e_J = np.abs(J_pred - J_test_)  # / J_test[test_ids].max()
        e_R = np.abs(R_pred - R_test_)  # / R_test_.max()
        e_B = np.abs(B_pred - B_test_)  # / B_test[test_ids].max()

        fig, axs = plt.subplots(3, 3)
        color_factor = 1
        fig.suptitle(f"Epoch {epoch_number}")
        im = []
        im.append(
            axs[0, 0].imshow(
                J_pred, vmin=color_factor * pred_min, vmax=color_factor * pred_max
            )
        )
        im.append(
            axs[1, 0].imshow(
                J_test_, vmin=color_factor * test_min, vmax=color_factor * test_max
            )
        )
        im.append(
            axs[2, 0].imshow(
                e_J,
                vmin=color_factor * error_min,
                vmax=color_factor * error_max,
                cmap=cmap_error,
            )
        )
        im.append(
            axs[0, 1].imshow(
                R_pred, vmin=color_factor * pred_min, vmax=color_factor * pred_max
            )
        )
        im.append(
            axs[1, 1].imshow(
                R_test_, vmin=color_factor * test_min, vmax=color_factor * test_max
            )
        )

        axs[2, 1].imshow(
            e_R,
            vmin=color_factor * error_min,
            vmax=color_factor * error_max,
            cmap=cmap_error,
        )

        im_pred = axs[0, 2].imshow(
            B_pred, vmin=color_factor * pred_min, vmax=color_factor * pred_max
        )

        im_test = axs[1, 2].imshow(
            B_test_, vmin=color_factor * test_min, vmax=color_factor * test_max
        )

        im_e = axs[2, 2].imshow(
            e_B,
            vmin=color_factor * error_min,
            vmax=color_factor * error_max,
            cmap=cmap_error,
        )

        im_cb_list = [im_pred, im_test, im_e]

        print(f"error_min: {error_min}, error_max: {error_max}")
        print(f"error_min: {test_min}, error_max: {test_max}")

        # common options for all axes
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        axs[0, 0].set_ylabel(r"$\tilde{\bm{J}}$")
        axs[1, 0].set_ylabel(r"$\hat{\bm{J}}$")
        axs[2, 0].set_ylabel(r"$\bm{J}_e$")
        axs[0, 1].set_ylabel(r"$\tilde{\bm{R}}$")
        axs[1, 1].set_ylabel(r"$\hat{\bm{R}}$")
        axs[2, 1].set_ylabel(r"$\bm{R}_e$")
        axs[0, 2].set_ylabel(r"$\tilde{\bm{B}}$")
        axs[1, 2].set_ylabel(r"$\hat{\bm{B}}$")
        axs[2, 2].set_ylabel(r"$\bm{B}_e$")

        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.subplot_tool()
        # plt.show()

        # cbar = fig.colorbar(im, ax=axs[0, 2], label="scaled values")
        # fig.colorbar(im, ax=axs[1, 2], label="scaled values")
        # fig.colorbar(im, ax=axs[2, 2], label="error")

        # cbar.get_position()

        # move B matrix to the left
        posB_x = axs[0, 2].get_position().x0  # Get the original position
        posR_x = axs[0, 1].get_position().x1
        gap = posB_x - posR_x  # Calculate the gap between the subplots
        desired_gap = 0.07
        transition = gap - desired_gap
        cbar_labels = ["norm. to orig.", "normalized", "rel. error"]
        for i in range(3):
            pos_ax = axs[i, 2].get_position()
            axs[i, 2].set_position(
                [pos_ax.x0 - transition, pos_ax.y0, pos_ax.width, pos_ax.height]
            )
            # add colorbar to the right
            l, b, w, h = axs[i, 2].get_position().bounds
            distance_to_cbar = 0.05
            cax = fig.add_axes([l + w + distance_to_cbar, b, distance_to_cbar, h])
            fig.colorbar(im_cb_list[i], cax=cax, label=cbar_labels[i])

        # ax2 = axs[0,2]
        # l, b, w, h = ax2.get_position().bounds         # get position of `ax2`
        # cax = fig.add_axes([l + w + 0.03, b, 0.03, h]) # add colorbar's axes next to `ax2`
        # fig.colorbar(im, cax=cax)

        # cbar = fig.colorbar(im, ax=axs[0, 2], label="scaled values")
        # fig.colorbar(im, ax=axs[1, 2], label="scaled values")
        # fig.colorbar(im, ax=axs[2, 2], label="error")

        # fig.tight_layout()

        # from mpl_toolkits.axes_grid1 import make_axes_locatable

        # divider = make_axes_locatable(axs[0, 2])
        # cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
        # fig.add_axes(cax)
        # fig.colorbar(im, cax=cax,orientation="vertical")

        # plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, f"matrices_for_gif_{epoch_number}.png"),
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=True,
            dpi=dpi,
        )
        plt.close()

        # create image list
        image_list.append(
            Image.open(os.path.join(result_dir, f"matrices_for_gif_{epoch_number}.png"))
        )

        del fig
        del axs
        del im

    logging.info(f"J_max_overall: {J_max_overall}")
    logging.info(f"R_max_overall: {R_max_overall}")
    logging.info(f"B_max_overall: {B_max_overall}")
    logging.info(f"J_min_overall: {J_min_overall}")
    logging.info(f"R_min_overall: {R_min_overall}")
    logging.info(f"B_min_overall: {B_min_overall}")

    # create gif from png files
    image_list[0].save(
        os.path.join(result_dir, "matrix_evolution_msd.gif"),
        save_all=True,
        append_images=image_list[1:],
        duration=duration,
        loop=loop,
        disposal=2,
    )


def normalize_to_range(arr, min_val=-1, max_val=1):
    original_min = np.min(arr)
    original_max = np.max(arr)

    # Normalize the array to [0, 1]
    normalized_arr = (arr - original_min) / (original_max - original_min)

    # Scale to [min_val, max_val]
    normalized_arr = normalized_arr * (max_val - min_val) + min_val

    return normalized_arr, original_min, original_max


def scale_array_with_orig_values(
    arr, original_min, original_max, min_val=-1, max_val=1
):
    # Normalize the array to [0, 1] using original min and max
    scaled_arr = (arr - original_min) / (original_max - original_min)

    # Scale to [min_val, max_val]
    scaled_arr = scaled_arr * (max_val - min_val) + min_val

    return scaled_arr


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def extract_matrices_and_errors(data, system_layer, test_ids, returnJRBminmax=True):
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

    e_J = np.abs(J_pred - J_test_)  # / J_test[test_ids].max()
    e_R = np.abs(R_pred - R_test_)  # / R_test_.max()
    e_B = np.abs(B_pred - B_test_)  # / B_test[test_ids].max()

    # get min max values
    J_min, J_max = min(J_pred.min(), J_test_.min()), max(J_pred.max(), J_test_.max())
    R_min, R_max = min(R_pred.min(), R_test_.min()), max(R_pred.max(), R_test_.max())
    B_min, B_max = min(B_pred.min(), B_test_.min()), max(B_pred.max(), B_test_.max())

    pred_min, pred_max = min(J_pred.min(), R_pred.min(), B_pred.min()), max(
        J_pred.max(), R_pred.max(), B_pred.max()
    )
    test_min, test_max = min(J_test_.min(), R_test_.min(), B_test_.min()), max(
        J_test_.max(), R_test_.max(), B_test_.max()
    )
    error_min, error_max = min(e_J.min(), e_R.min(), e_B.min()), max(
        e_J.max(), e_R.max(), e_B.max()
    )

    if returnJRBminmax:
        return (
            J_pred,
            R_pred,
            B_pred,
            J_test_,
            R_test_,
            B_test_,
            J_min,
            J_max,
            R_min,
            R_max,
            B_min,
            B_max,
            e_J,
            e_R,
            e_B,
        )
    else:
        return (
            J_pred,
            R_pred,
            B_pred,
            J_test_,
            R_test_,
            B_test_,
            pred_min,
            pred_max,
            test_min,
            test_max,
            error_min,
            error_max,
            e_J,
            e_R,
            e_B,
        )
