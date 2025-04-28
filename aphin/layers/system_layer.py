"""
TensorFlow layers for identification of dynamical systems
e.g. LTI systems (LTILayer), port-Hamiltonian systems (PHLayer), ...
intended for standalone use or the use in the latent space of an autoencoder.
All layers approximate the right-hand side of the ODE given states z(t), inputs u(t) and parameters mu
and should be trained using reference values of the states, inputs, parameters and the left-hand side of the ODE
which is assumed to depend only on the time derivative of the states.
"""

from abc import abstractmethod
import numpy as np
import tensorflow as tf


class SystemLayer(tf.keras.layers.Layer):
    """
    Layer for identification of an abstract system z'(t) = f(z, u, mu).

    This layer approximates the right-hand side of the ODE given states z(t), inputs u(t), and parameters mu.
    It is intended for standalone use or for use in the latent space of an autoencoder.
    """

    def __init__(
        self, r, n_u=None, n_mu=None, regularizer=None, dtype=tf.float32, **kwargs
    ):
        """
        Initialize the SystemLayer.

        Parameters
        ----------
        r : int
            Number of latent variables.
        n_u : int, optional
            Number of inputs, by default None.
        n_mu : int, optional
            Number of parameters, by default None.
        regularizer : tf.keras.regularizers.Regularizer, optional
            Regularizer object for trainable variables of the class, by default None.
        dtype : tf.DType, optional
            Data type, by default tf.float32.
        **kwargs : dict
            Additional arguments for the tf.keras.layers.Layer base class.
        """
        super(SystemLayer, self).__init__(dtype=dtype, **kwargs)
        self.dtype_ = dtype
        self.r = r
        self.n_u = 0 if n_u is None else n_u
        self.n_mu = 0 if n_mu is None else n_mu
        self.regularizer = regularizer

    @abstractmethod
    @tf.function
    def call(self, z, u=None, mu=None, training=False):
        """
        Evaluate rhs of the ODE z'(t) = f(z, u, mu) for inputs (z, u, mu).

        Parameters
        ----------
        z : array-like
            State variables.
        u : array-like, optional
            Input variables, by default None.
        mu : array-like, optional
            Parameter variables, by default None.
        training : bool, optional
            Whether the call is in training mode, by default False.

        Returns
        -------
        tf.Tensor
            Approximation of z'(t).
        """
        pass

    def lhs(self, dz_dt):
        """
        Evaluate left-hand side of the system ODE given z'(t) as dz_dt.

        For descriptor systems: E * z'(t).

        Parameters
        ----------
        dz_dt : array-like
            Time derivative of the system states z'(t).

        Returns
        -------
        tf.Tensor
            Left-hand side of the system ODE.
        """
        dz_dt = tf.cast(dz_dt, dtype=self.dtype_)
        return tf.identity(dz_dt)

    def get_config(self):
        """
        Get the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()
        config.update({"regularizer": self.regularizer})
        return config
