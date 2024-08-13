# default packages
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import logging
import matplotlib.pyplot as plt

# own packages
from aphin import config
from examples.examples_utils.pixel_shape import (
    create_pixel_circle,
    create_pixel_rectangle,
)


def gen_video_data(
    example,
    data_path,
    save_path=None,
    video_name=None,
    mass_shape="circle",
    create_test_data=True,
    num_pixels=np.array([32, 32]),
    scale_pixel=None,
    skip_time_step_factor=None,
    size=None,
):
    """
    Generates pixel grey value video data from precalculated trajectories of a mass-spring-damper or pendulum
    :param example: 'msd' or 'pendulum' - example that will be used
    :param data_path: data path to the precalculated trajectory
    :param save_path: {None} save path to save the .npz file, if None use cwd
    :param video_name: {None} choose a name for the video, if None it uses f'{example}_{mass_shape}'
    :param mass_shape: {'circle'} or 'rectangle' shape of the bodies
    :param create_test_data: {True} creates test video data
    :param num_pixels: {'[32 32]'} np.array which states the frame size
    :param scale_pixel: {None} if 1 that trajectory size is used as pixel unit, e.g. 1m = 1pixel, if None it uses default values {20} for the 'msd' example and {4} for 'pendulum' example
    :param skip_time_step_factor (int): {'None'} use every skip_time_step of the time trajectory, if None it uses default values {4} for the 'msd' and 'pendulum' example
    :param size (int or array): {None} size of the object, e.g. radius of the circle or array for rectangle size, if None it uses default values {2.5} for the 'msd' and {2} for 'pendulum' example and circle mass shape
    :return:
        pixel_data: saved as file '{video_name}.npz' and '{video_name}_test.npz' that includes pixel_frames, input u, parameter mu and parameter_information which is taken from the data
    """

    # scale value and skip time factor
    if example == "msd":
        if scale_pixel is None:
            scale_pixel = 20
        if skip_time_step_factor is None:
            skip_time_step_factor = (
                4  # use every skip_time_step_factor value of time trajectory
            )
    elif example == "pendulum":
        if scale_pixel is None:
            scale_pixel = 4
        if skip_time_step_factor is None:
            skip_time_step_factor = (
                4  # use every skip_time_step_factor value of time trajectory
            )

    # radius size
    if mass_shape == "circle":
        if size is None:
            if example == "msd":
                radius = 2.5
            elif example == "pendulum":
                radius = 2
        else:
            radius = size
    elif mass_shape == "rectangle":
        if size is None:
            rectangle_pixel_size = np.array([10, 7.5])
        else:
            rectangle_pixel_size = size
    else:
        raise ValueError(f"mass_shape {mass_shape} not known")

    # video name default
    if video_name is None:
        video_name = f"{example}_{mass_shape}"

    if save_path is None:
        save_path = os.getcwd()  # current working directory

    # set up logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    # set up matplotlib
    plt.rcParams.update(
        {
            "text.usetex": True,
        }
    )

    # %% load msd data
    data = np.load(os.path.join(data_path))
    t = data["t"]
    x = data["states"].transpose([2, 0, 1])

    u = data["u"].transpose([2, 0, 1])
    if example == "msd":
        mu = data["sampled_parameters"][:, :3]
    elif example == "pendulum":
        # only initial values
        mu = data["phi_0"]

    parameter_information = data["parameter_information"]
    # numerically differentiate states to obtain time derivatives
    dx_dt = np.gradient(x, t[1] - t[0], axis=1)
    # repeat mu for each sample
    mu = np.repeat(mu[:, np.newaxis], x.shape[1], axis=1)
    if np.ndim(mu) == 2:
        mu = np.expand_dims(mu, axis=2)

    n_u = u.shape[2]

    # system dimensions
    n_sim = x.shape[0]
    n_t = x.shape[1]
    n = x.shape[2]
    n_mu = mu.shape[2]

    # scale each axis of mu to [-1, 1]
    mu_min = np.min(mu, axis=(0, 1))
    mu_max = np.max(mu, axis=(0, 1))
    mu = 2 * (mu - mu_min) / (mu_max - mu_min) - 1

    # scale each axis of x to [-1, 1]
    # x_min = np.min(x, axis=(0, 1))
    # x_max = np.max(x, axis=(0, 1))
    # x = 2*(x - x_min)/(x_max - x_min) - 1

    # split data into training and validation set
    train_test_split_ratio = 0.8
    train_ids = np.arange(int(n_sim * train_test_split_ratio))
    test_ids = np.arange(int(n_sim * train_test_split_ratio), n_sim)

    x_train, x_test, dx_dt_train, dx_dt_test, mu_train, mu_test, u_train, u_test = (
        train_test_split(
            x, dx_dt, mu, u, train_size=train_test_split_ratio, shuffle=True
        )
    )

    n_test = x_test.shape[0]
    # # get train and test ids
    # train_ids = np.arange(x_train.shape[0])
    # test_ids = np.arange(x_test.shape[0])

    # reshape data to fit into model
    x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1], n)
    u_train = u_train.reshape(u_train.shape[0] * u_train.shape[1], n_u)
    mu_train = mu_train.reshape(mu_train.shape[0] * mu_train.shape[1], n_mu)
    x_test = x_test.reshape(x_test.shape[0] * x_test.shape[1], n)
    u_test = u_test.reshape(u_test.shape[0] * u_test.shape[1], n_u)
    mu_test = mu_test.reshape(mu_test.shape[0] * mu_test.shape[1], n_mu)

    # skip some time steps
    x_train = x_train[::skip_time_step_factor, :]
    u_train = u_train[::skip_time_step_factor, :]
    mu_train = mu_train[::skip_time_step_factor, :]
    x_test = x_test[::skip_time_step_factor, :]
    u_test = u_test[::skip_time_step_factor, :]
    mu_test = mu_test[::skip_time_step_factor, :]

    def initialize_pixel_array(num_pixels=None):
        color_white = 255
        # default values
        if num_pixels is None:
            num_pixels = np.array([100, 100])

        # initialize pixel array with white
        pixel_array = np.ones((num_pixels)) * color_white
        return pixel_array

    # scale data
    x_train = scale_pixel * x_train
    x_test = scale_pixel * x_test

    # %% call pixel function
    def create_frames(x, num_pixels, mass_shape, radius=None, y=None):

        # we interpret the grid as coordinate system and define the start value of each mass
        # at equidisant points on the grid in x and centered in y
        n_mass = int(x.shape[1] / 2)

        pos_y_offset = np.ones(n_mass) * num_pixels[1] / 2
        pos_x_offset = (np.arange(n_mass) + 1) * (num_pixels[0] / (n_mass + 1))

        # validate data against image border
        min_first_mass_x = np.min(x[:, 0])
        max_last_mass_x = np.max(x[:, n_mass - 1])
        if y is not None:
            min_first_mass_y = np.min(y[:, 0])
            max_last_mass_y = np.max(y[:, n_mass - 1])
        if mass_shape == "circle":
            most_left_value_x = pos_x_offset[0] + min_first_mass_x - np.ceil(radius)
            most_right_value_x = pos_x_offset[-1] + max_last_mass_x + np.ceil(radius)
            if y is not None:
                most_left_value_y = pos_y_offset[0] + min_first_mass_y - np.ceil(radius)
                most_right_value_y = (
                    pos_y_offset[-1] + max_last_mass_y + np.ceil(radius)
                )
                if most_left_value_y < 0 or most_right_value_y > num_pixels[1]:
                    raise ValueError(
                        "Masses will cross the image border in y-direction. Please increase the number of pixel in y or decrease the pixel scale factor."
                    )

        if most_left_value_x < 0 or most_right_value_x > num_pixels[0]:
            raise ValueError(
                "Masses will cross the image border in x-direction. Please increase the number of pixel in x or decrease the pixel scale factor."
            )

        # validate data - masses should not collide
        if n_mass > 1:
            x_in_pixel_coord = x[:, :n_mass] + pos_x_offset
            diff_x_in_pixel = np.diff(x_in_pixel_coord, axis=1)
            if mass_shape == "circle":
                if (
                    np.min(diff_x_in_pixel) < 2 * np.ceil(radius) + 1
                ):  # minimum distance of 1 pixel
                    raise ValueError(
                        "Masses will collide. Please increase the number of pixel in x or decrease the pixel scale factor."
                    )

        frames = []
        center_point = np.zeros(2)
        pixel_frames = np.zeros((num_pixels[1], num_pixels[0], x.shape[0]))
        for time_idx, x_value in enumerate(x):
            if np.floor((time_idx - 1) / x.shape[0] * 10) < np.floor(
                (time_idx) / x.shape[0] * 10
            ):
                # print progress in multiples of 10 %
                logging.info(
                    f"Processed {np.floor((time_idx)/x.shape[0]*10)*10}% of the data to pixel frames."
                )

            # reset pixel_array to all white
            pixel_array = initialize_pixel_array(num_pixels=num_pixels)
            for i_mass in range(n_mass):
                center_point[0] = pos_x_offset[i_mass] + x[time_idx, i_mass]
                if y is not None:
                    center_point[1] = pos_y_offset[i_mass] + y[time_idx, i_mass]
                else:
                    center_point[1] = pos_y_offset[i_mass]
                if mass_shape == "circle":
                    pixel_array = create_pixel_circle(
                        center_point, pixel_array, r=radius
                    )
                elif mass_shape == "rectangle":
                    pixel_array = create_pixel_rectangle(
                        center_point, pixel_array, r=radius
                    )
            frames.append(Image.fromarray(pixel_array.T).convert("L"))
            pixel_frames[:, :, time_idx] = pixel_array.T

        return frames, pixel_frames

    if example == "msd":
        x_coordinate_train = x_train
        y_coordinate_train = None
    elif example == "pendulum":
        x_coordinate_train = x_train[:, (0, 2)]
        y_coordinate_train = x_train[:, (1, 3)]
        plot_traj = True
        if plot_traj:
            plt.figure()
            plt.plot(x_coordinate_train[:, 0], y_coordinate_train[:, 0], "cyan")
            plt.ylabel("y")
            plt.xlabel("x")
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            plt.show(block=False)

    frames, pixel_frames = create_frames(
        x_coordinate_train, num_pixels, mass_shape, radius=radius, y=y_coordinate_train
    )
    logging.info(f"Saving gif: {video_name}.gif .")
    frames[0].save(
        os.path.join(save_path, f"{video_name}.gif"),
        append_images=frames[1:],
        save_all=True,
        duration=50,  # time of each frame in ms
        loop=0,  # loops for ever
    )

    logging.info(f"Saving pixel data to {video_name}.npz...")
    np.savez_compressed(
        os.path.join(save_path, f"{video_name}.npz"),
        pixel_frames=pixel_frames,
        u=u_train,
        mu=mu_train,
        parameter_information=parameter_information,
    )

    # %% test/validation data
    if create_test_data:
        if example == "msd":
            x_coordinate_test = x_test
            y_coordinate_test = None
        elif example == "pendulum":
            x_coordinate_test = x_test[:, (0, 2)]
            y_coordinate_test = x_test[:, (1, 3)]
        frames_test, pixel_frames_test = create_frames(
            x_coordinate_test,
            num_pixels,
            mass_shape,
            radius=radius,
            y=y_coordinate_test,
        )
        logging.info(f"Saving test gif: {video_name}_test.gif .")
        frames_test[0].save(
            os.path.join(save_path, f"{video_name}_test.gif"),
            append_images=frames_test[1:],
            save_all=True,
            duration=50,  # time of each frame in ms
            loop=0,  # loops for ever
        )

        logging.info(f"Saving pixel test data to {video_name}_test.npz...")
        np.savez_compressed(
            os.path.join(save_path, f"{video_name}_test.npz"),
            pixel_frames=pixel_frames_test,
            u=u_test,
            mu=mu_test,
            parameter_information=parameter_information,
        )

    print("debug stop")
