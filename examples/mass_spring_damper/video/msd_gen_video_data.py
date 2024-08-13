from examples.examples_utils.gen_video_data import gen_video_data
import numpy as np
import os


example = "msd"
data_path = f"{os.path.dirname(__file__)}/../data/Halton_state_dim5_samples_120.npz"
save_path = f"{os.path.dirname(__file__)}/../data/"
mass_shape = "circle"
video_name = f"{example}_{mass_shape}"
create_test_data = True
num_pixels = np.array([32, 32])
scale_pixel = 2
skip_time_step_factor = 4
size = 1.5  # radius

if not os.path.exists(save_path):
    os.makedirs(save_path)

gen_video_data(
    example,
    data_path,
    save_path,
    video_name,
    mass_shape,
    create_test_data,
    num_pixels,
    scale_pixel,
    skip_time_step_factor,
    size,
)
