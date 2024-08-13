"""
Model order reduction of a disc brake model using port-Hamiltonian autoencoders
"""

# default packages
import numpy as np
import logging
import datetime
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer

# third party packages
import tensorflow as tf
import tensorflow.keras.backend as K

# own packages
from aphin import config
from aphin.identification import APHIN, ConvAPHIN
from aphin.layers import PHLayer, PHQLayer
from aphin.utils.visualizations import setup_matplotlib
from aphin.systems import PHSystem
from matplotlib.animation import FuncAnimation

# set up logging
tf.config.run_functions_eagerly(True)  # uncomment this line for debugging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# set up matplotlib
save_plots = False
setup_matplotlib(save_plots)


def animate_approximation(x, x_rec, error):

    n_samples = x.shape[0]
    # plot comparison of original and reconstructed data as animation

    fig, ax = plt.subplots(
        1, 3, figsize=(5.7, 3.0), dpi=300, sharex="all", sharey="all"
    )
    im1 = ax[0].imshow(x[0], cmap="gray")
    ax[0].set_title("Original")
    im2 = ax[1].imshow(x_rec[0], cmap="gray")
    ax[1].set_title("Reconstructed")
    im3 = ax[2].imshow(error[0], cmap="jet", vmin=-x.max(), vmax=x.max())
    # fig.colorbar(im, cax=ax[2], orientation='vertical')
    ax[2].set_title("error")
    # add colorbar to last subplot
    fig.colorbar(im3, ax=ax[2], orientation="vertical")

    # initialization function: plot the background of each frame
    def init():
        im1.set_data(x[0])
        im2.set_data(x_rec[0])
        im3.set_data(error[0])
        return [im1]  # , im2, im3

    # animation function.  This is called sequentially
    def animate(i):  # exponential decay of the values
        # print(i)
        im1.set_array(x[i])
        im2.set_array(x_rec[i])
        im3.set_array(error[i])
        return [im1]  # , im2, im3

    plt.show()
    ani = FuncAnimation(
        fig, animate, frames=range(0, n_samples), init_func=init, blit=True
    )

    # save animation as gif
    # To save the animation using Pillow as a gif
    import matplotlib.animation as animation

    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save("video_data.gif", writer=writer)


# %% Script parameters
# Save model information for analysis in tensorboard. Use ```tensorboard --logdir logs/fit``` to view the results
model = "pendulum"  # msd, pendulum
if model == "pendulum":
    r = 2
elif model == "msd":
    r = 6
nth_step = 2
dt = 0.04 * nth_step
test_name = "conv"  # "regular", "conv"

output_dir = os.path.join("../results", model)
# Determine paths
datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(output_dir, "logs/" + datetime_str)
weight_dir = os.path.join(output_dir, "weights/", f"{test_name}.weights.h5")
video_path = config.video_path

# %% Read simulated disc brake data
logging.info(
    "################################   1. Data       ################################"
)
data_path = os.path.join(video_path, f"{model}_circle.npz")
test_data_path = os.path.join(video_path, f"{model}_circle_test.npz")


# load data
train_data, test_data = dict(path=data_path), dict(path=test_data_path)
for data_dict in [train_data, test_data]:
    with open(data_dict["path"], "rb") as f:
        data = np.load(f)

        data_dict["pixels"] = data["pixel_frames"].transpose([2, 0, 1])[::nth_step]
        # normalize data to [0, 1]
        data_dict["pixels"] = data_dict["pixels"] / 255
        if test_name != "conv":
            data_dict["pixels"] = data_dict["pixels"].reshape(
                [data_dict["pixels"].shape[0], -1]
            )
        data_dict["pixels_dt"] = np.gradient(data_dict["pixels"], dt, axis=0)
        data_dict["u"] = data["u"][::nth_step]
        data_dict["mu"] = data["mu"][::nth_step]

x, dxdt, u, mu = (
    train_data["pixels"],
    train_data["pixels_dt"],
    train_data["u"],
    train_data["mu"],
)
x_test, dxdt_test, u_test, mu_test = (
    test_data["pixels"],
    test_data["pixels_dt"],
    test_data["u"],
    test_data["mu"],
)
n_s, n_t = 1, x.shape[0]

# %% Create APHIN
logging.info(
    "################################   2. Model      ################################"
)
# Model parameters
use_pca = False
pca_scaling = True
# Loss parameters
l_rec = K.variable(1.0)
l_dz = K.variable(1.0)
l_dx = K.variable(0 * 1e-2)
# convolutional parameters
n_filters = [2, 4, 6]
kernel_size = [8, 4, 2]
strides = [2, 2, 2]
# Regularization parameters
l1 = 0
l2 = 0
# Training parameters
n_epochs = 5000
batch_size = 32
learning_rate = 0.00025
use_Q_matrix = True
load_network = False

# Path to weights sub folder
# Check if folder exists and create if it does not exist
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        weight_dir,
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
    ),
    # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
]
callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
if use_Q_matrix:
    system_layer = PHQLayer(
        r,
        n_mu=mu.shape[1],
        n_u=u.shape[1],
        regularizer=regularizer,
        layer_sizes=[32, 32],
    )
else:
    system_layer = PHLayer(
        r,
        n_mu=mu.shape[1],
        n_u=u.shape[1],
        regularizer=regularizer,
        layer_sizes=[32, 32],
    )
# system_layer = ThermoMechPHLayer(r, n_u=u.shape[1])

if test_name == "conv":
    aphin = ConvAPHIN(
        r,
        n_filters,
        kernel_size,
        strides,
        x=x,
        u=u,
        mu=mu,
        system_layer=system_layer,
        activation="elu",
        use_pca=use_pca,
        pca_scaling=pca_scaling,
        l_rec=l_rec,
        l_dz=l_dz,
        l_dx=l_dx,
    )
else:
    aphin = APHIN(
        r,
        x=x,
        u=u,
        mu=mu,
        system_layer=system_layer,
        layer_sizes=[32, 16],
        activation="elu",
        pca_order=64,
        use_pca=True,
        pca_scaling=pca_scaling,
        l_rec=l_rec,
        l_dz=l_dz,
        l_dx=l_dx,
    )

# %%
aphin.compile(optimizer="adam", loss=tf.keras.losses.MSE)
load_network = False
if load_network:
    aphin.load_weights(weight_dir)
    logging.info(f"Loaded weights from file {weight_dir}")
    aphin.save_weights(weight_dir)
else:
    # phin.load_weights(data_path_weights_filename+"_10k")
    # Fit model
    # with tf.device('/device:CPU:0'):
    # model.fit(x_train, y_train)
    train_hist = aphin.fit(
        x=[x, dxdt, u, mu],
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    # plot loss
    fig, ax = plt.subplots(1, 3, figsize=(5.7, 3.0), dpi=300, sharex="all")
    ax[0].semilogy(train_hist.history["loss"])
    ax[0].grid(linestyle=":", linewidth=1)
    ax[0].set_xlabel("Epoch")
    ax[0].set_title("Loss")
    ax[1].semilogy(train_hist.history["rec_loss"])
    ax[1].grid(linestyle=":", linewidth=1)
    ax[1].set_xlabel("Epoch")
    ax[1].set_title("Reconstruction loss")
    ax[2].semilogy(train_hist.history["dz_loss"])
    ax[2].grid(linestyle=":", linewidth=1)
    ax[2].set_xlabel("Epoch")
    ax[2].set_title("dz loss")
    plt.tight_layout()

# %% Animation reconstruct data
x_rec = aphin.reconstruct(x).numpy().squeeze()
diff = x - x_rec
error = np.abs(diff)
animate_approximation(
    x[100:200:3].reshape(-1, 32, 32),
    x_rec[100:200:3].reshape(-1, 32, 32),
    diff[100:200:3].reshape(-1, 32, 32),
)

# %% Validation of time derivatives
z, dz_dt = aphin.calc_latent_time_derivatives(x_test, dxdt_test)
# x_rec_, dx_dt_rec = phin.calc_physical_time_derivatives(z, dz_dt)
dz_dt_rec = aphin.system_network([z, u_test, mu_test])
dz_dt_finite_diff = np.gradient(z, dt, axis=0)
# %% Validation of time derivatives
z, dz_dt = aphin.calc_latent_time_derivatives(x, dxdt)
# x_rec_, dx_dt_rec = phin.calc_physical_time_derivatives(z, dz_dt)
dz_dt_rec = aphin.system_network([z, u, mu])

dz_dt_finite_diff = np.gradient(z, dt, axis=0)


# plot dzdt
i = 0
plt.figure()
plt.plot(dz_dt[:, i])
plt.plot(dz_dt_rec[:, i])
plt.plot(dz_dt_finite_diff[:, i])
plt.legend(["$\dot{z}$", "$\dot{z}_{ph}$", "$\dot{z}_{fd}$"])
plt.show(block=False)

plt.figure()
plt.plot(z[:, i])
plt.legend(["$z$"])
plt.show(block=False)

# %% Validation of reconstruction
rel_error = np.mean(np.linalg.norm(error, axis=(1)) / np.linalg.norm(x, axis=(1)))
rel_error = np.mean(np.linalg.norm(error, axis=(1, 2)) / np.linalg.norm(x, axis=(1, 2)))

# %% Validation
# logging.info('################################   3. Validation ################################')

# Identified matrices for the ODE in latent coordinates:
J_ph, R_ph, B_ph, Q_ph = system_layer.get_system_matrices(mu_test)
np.set_printoptions(formatter={"float": "\t{: 0.4f}\t".format})
logging.info(
    f"Identified matrices for the ODE in latent coordinates: \n "
    f"__________ J_ph: __________\n {J_ph}\n"
    f"__________ R_ph: __________\n {R_ph}\n"
    f"__________ Q_ph: __________\n {Q_ph}\n"
    f"__________ B_ph: __________\n {B_ph}"
)
logging.info(f"Eigenvalues of R_ph: {np.linalg.eigvals(R_ph)}")

A = (J_ph - R_ph) @ Q_ph


# %% Validation of the pH approximation
i_sim = 4
n_t = 50
x_test_ = x_test[i_sim * n_t : (i_sim + 1) * n_t]
u_test_ = u_test[i_sim * n_t : (i_sim + 1) * n_t]
mu_test_ = mu_test[i_sim * n_t : (i_sim + 1) * n_t]
J_ph_, R_ph_, B_ph_, Q_ph_ = system_layer.get_system_matrices(mu_test_[0:1])
z_test_ = aphin.encode(x_test_).numpy().T
x_init = x_test_[0:1]
t = np.linspace(0, dt * n_t, n_t)
# Solve ODE system in latent space
ph_system = PHSystem(J_ph[i_sim], R_ph[i_sim], B=B_ph[i_sim], Q_ph=Q_ph[i_sim])
prediction_start = timer()
z_init = aphin.encode(x_init).numpy().T
z_ph, dz_dt_ph = ph_system.solve_dt(t, z_init, u_test[i_sim * n_t : (i_sim + 1) * n_t])
x_ph = aphin.decode(z_ph).numpy().squeeze()

fig, axs = plt.subplots(r, 1, sharex=True)
for i, (z_ph_, z_ref) in enumerate(zip(z_ph.squeeze().T, z_test_)):
    axs[i].plot(z_ref)
    axs[i].plot(z_ph_, "--")

# Reshape to (n_t * n_s, n)
z_ph_rs = z_ph.transpose([2, 0, 1]).reshape([-1, r])
dz_dt_ph_rs = dz_dt_ph.transpose([2, 0, 1]).reshape([-1, r])

# reconstruct images from latent state
x_ph = aphin.decode(z_ph_rs).numpy().squeeze()
diff = x_test_ - x_ph
error = np.abs(diff)
animate_approximation(x_test_, x_ph, diff)
