# default packages
import numpy as np
import logging
import datetime
import matplotlib.pyplot as plt

# third party packages
import tensorflow as tf

# own packages
from aphin.identification import APHIN
from aphin import config
from aphin.utils.visualizations import setup_matplotlib

# set up logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# tf.config.run_functions_eagerly(True)

# set up matplotlib
save_plots = False
setup_matplotlib(save_plots)

# %% Script parameters
tensorboard = False
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_path = config.data_path  # dummy variable to showcase the use of config.yml

# %% Create dummy data
logging.info(
    "################################   1. Data       ################################"
)
n = 500  # number of features / DOF
# input data
n_t = 1000  # number of time steps / samples
t = np.linspace(0, 10, n_t)[:, np.newaxis]
# high-dimensional output data, we use u(t) = exp(w*t) * u0 which is the solution of the ODE u′ = -wu
weight = -2
x = (
    np.exp(weight * t) * 1 * np.random.normal(0, 0.1, n)[np.newaxis]
)  # each feature is a solution of the ODE
# add some noise to the data
x += np.random.normal(0, 0.0005, (n_t, n))
print(x.shape)
# calculate time derivative of x numerically
dx_dt = np.gradient(x, t[:, 0], axis=0)
print(dx_dt.shape)
# plot single feature with its time derivative
plt.figure()
plt.plot(t, dx_dt[:, 0], "cyan")
plt.plot(t, x[:, 0], "k")
plt.legend([r"$\dot{x}^{(0)}(t)$", r"$x^{(0)}(t)$"])
plt.show(block=False)
logging.info(
    f"Created noisy and {n}-dimensional data of the ODE u′ = -{weight}u with {n_t} samples"
)


# %% Create APHIN
logging.info(
    "################################   2. Model      ################################"
)
# create model
r = 1
ph_autoencoder = APHIN(
    r,
    x=x,
    mu=None,
    layer_sizes=[32],
    activation="selu",
    use_pca=True,
    pca_order=10,
    pca_scaling=False,
    l_rec=1,
    l_dz=1,
    l_dx=1,
)
# compile model
ph_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025))
# fit model
callbacks = []
if tensorboard:  # use ``` tensorboard --logdir logs/fit    ``` to view the results
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
train_hist = ph_autoencoder.fit(
    x=[x, dx_dt], epochs=1000, batch_size=64, verbose=2, callbacks=callbacks
)
logging.info(f"Created PHAutoencoder with {r} latent variables")
ph_autoencoder.vis_modes(x, 3)

# %% Validation
logging.info(
    "################################   3. Validation ################################"
)

# compare state reconstruction
idx = 1
x_rec = ph_autoencoder.reconstruct(x)
# plot selected state
plt.figure()
plt.plot(t, x[:, idx], "cyan")  # real data
plt.plot(t, x_rec[:, idx], "magenta")  # reconstructed data
x_legend, dx_dt_legend = rf"$x_{idx}(t)$", r"$\tilde{x}" + f"_{idx}(t)$"
plt.legend([x_legend, dx_dt_legend])
plt.show()

# compare latent variables
idx = 0
# calculate latent time derivatives
z, dz_dt = ph_autoencoder.calc_latent_time_derivatives(x, dx_dt)
# predicted time derivatives
dz_dt_ph = ph_autoencoder.system_network(tf.concat([z], axis=1))
# plot latent variables
plt.figure()
plt.plot(t, z[:, idx], "k")  # real data
plt.plot(t, dz_dt[:, idx], "cyan")  # real data
plt.plot(t, dz_dt_ph[:, idx], "magenta", linestyle="dashed")  # reconstructed data
plt.legend([r"$z$", r"$\dot{z}$", r"$\dot{\tilde{z}}$"])
plt.show()


# identified weight for the latent ode (should be close to weight)
R_ph = ph_autoencoder.system_network.layers[1].R.to_dense()
weight_ph = -R_ph.numpy()[0, 0]
logging.info(
    f"Identified weight for the latent ODE: {weight_ph}"
    f" (should be close to {weight})"
)

x_init = x[0, :]
x_ph = tf.exp(weight_ph * t) * x_init
x_true = tf.exp(weight * t) * x_init
plt.figure()
plt.plot(tf.squeeze(t), x_ph[:, 0])
plt.plot(tf.squeeze(t), x_true[:, 0], "--")
plt.legend([r"$x_1(t)$", r"$x_1(t)$"])
plt.show(block=False)
