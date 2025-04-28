from abc import ABC
import logging
import tensorflow as tf
from matplotlib import pyplot as plt

from . import PHBasemodel
from aphin.layers import PHLayer

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class PHIN(PHBasemodel):
    """
    port-Hamiltonian identification network (phin).
    Model to discover the dynamics of a system using a layer for identification of other dynamical systems
    (see SystemLayer), e.g., a PHLayer (port-Hamiltonian).
    """

    def __init__(
        self,
        reduced_order,
        x=None,
        u=None,
        mu=None,
        system_layer=None,
        dtype="float32",
        **kwargs,
    ):
        """
        initialization of the PHIN model

        Parameters
        ----------
        reduced_order : int
            Order of the reduced model.
        x : array-like, optional
            Input data (full states) with shape (n_t * n_s, n) with n states, n_t time steps, n_s simulations.
        u : array-like, optional
            System inputs with shape (n_t * n_s, n_u) with n_u inputs, n_t time steps, n_s simulations.
        mu : array-like, optional
            Parameter data of shape (n_t * n_s, n_p) with n_p parameters, n_t time steps, n_s simulations.
        system_layer : SystemLayer, optional
            SystemLayer (or subclass) instance that learns the reduced system in the latent space.
        dtype : str, optional
            Data type, by default "float32".
        **kwargs : dict
            Additional arguments for the PHBasemodel base class.
        """
        self.dtype_ = dtype
        super(PHIN, self).__init__(**kwargs)

        if not hasattr(self, "config"):
            self._init_to_config(locals())

        # general parameters
        self.system_optimizer = None
        self.reduced_order = reduced_order
        if system_layer is None:
            self.system_layer = PHLayer(self.reduced_order)
        else:
            self.system_layer = system_layer

        # create the model
        x = tf.cast(x, dtype=self.dtype)
        self.x_shape = x.shape[1:]
        if u is not None:
            u = tf.cast(u, dtype=self.dtype)
            self.u_shape = u.shape[1:]
        if mu is not None:
            mu = tf.cast(mu, dtype=self.dtype)
        self.build_model(x, u, mu)

        # create loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dz_loss_tracker = tf.keras.metrics.Mean(name="dz_loss")
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")

    def build_model(self, x, u, mu):
        """
        Build the model.

        Parameters
        ----------
        x : array-like
            Full state with shape (n_samples, n_features).
        u : array-like, optional
            Inputs with shape (n_samples, n_inputs).
        mu : array-like, optional
            Parameters with shape (n_samples, n_params).

        Returns
        -------
        None
        """
        x_input = tf.keras.Input(shape=(x.shape[1],))

        # System inputs
        if u is not None:
            u_input = tf.keras.Input(shape=(u.shape[1],))
        else:
            u_input = tf.keras.Input(shape=(0,))

        # Simulation parameters
        if mu is not None:
            mu_input = tf.keras.Input(shape=(mu.shape[1],))
        else:
            mu_input = tf.keras.Input(shape=(0,))

        dx_dt_system = self.system_layer(x_input, u_input, mu_input)

        # network for discovery of dynamic system (e.g. pH system)
        self.system_network = tf.keras.Model(
            inputs=[x_input, u_input, mu_input], outputs=dx_dt_system, name="system"
        )

    @tf.function
    def train_step(self, inputs):
        """
        Perform one training step.

        Parameters
        ----------
        inputs : list of array-like
            Input data.

        Returns
        -------
        dict
            Dictionary containing the loss, dz_loss, and reg_loss.
        """

        # perform forward pass, calculate loss and update weights
        dz_loss, reg_loss, loss = self.build_loss(inputs)

        # update loss tracker
        self.loss_tracker.update_state(loss)
        self.dz_loss_tracker.update_state(dz_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {
            "loss": self.loss_tracker.result(),
            "dz_loss": self.dz_loss_tracker.result(),
            "reg_loss": self.reg_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, inputs):
        """
        Perform one test step.

        Parameters
        ----------
        inputs : list of array-like
            Input data.

        Returns
        -------
        dict
            Dictionary containing the loss, dz_loss, and reg_loss.
        """
        dz_loss, reg_loss, loss = self.build_loss(inputs)
        self.loss_tracker.update_state(loss)
        self.dz_loss_tracker.update_state(dz_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {
            "loss": loss,
            "dz_loss": dz_loss,
            "reg_loss": reg_loss,
        }

    def get_loss(self, x, dx_dt, u=None, mu=None):
        """
        Calculate loss.

        Parameters
        ----------
        x : array-like
            Full state with shape (n_samples, n_features).
        dx_dt : array-like
            Time derivative of state with shape (n_samples, n_features).
        u : array-like
            System input with shape (n_samples, n_inputs).
        mu : array-like, optional
            System parameters with shape (n_samples, n_parameters), by default None.

        Returns
        -------
        tuple
            Tuple containing dz_loss, reg_loss, and total loss.
        """

        # calculate left hand side of ODE system (relevant for descriptor systems)
        dx_dt_lhs = self.system_layer.lhs(dx_dt)

        # system_network approximation of the time derivative of the latent variable
        if mu is None:
            mu = tf.zeros([tf.shape(x)[0], 0])
        if u is None:
            u = tf.zeros([tf.shape(x)[0], 0])
        dx_dt_system = self.system_network([x, u, mu])

        dz_loss = self.compute_loss(None, dx_dt_lhs, dx_dt_system)

        # add regularization losses
        if self.system_network.losses:
            reg_loss = tf.math.add_n(self.system_network.losses)
        else:
            reg_loss = 0

        loss = dz_loss + reg_loss
        return dz_loss, reg_loss, loss
