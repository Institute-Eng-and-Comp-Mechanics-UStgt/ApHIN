"""
TensorFlow layers for identification of dynamical systems
e.g. LTI systems (LTILayer), port-Hamiltonian systems (PHLayer), ...
intended for standalone use or the use in the latent space of an autoencoder.
All layers approximate the right-hand side of the ODE given states z(t), inputs u(t) and parameters mu
and should be trained using reference values of the states, inputs, parameters and the left-hand side of the ODE
which is assumed to depend only on the time derivative of the states.
"""

import tensorflow as tf
from tensorflow.python.ops.linalg.linear_operator_full_matrix import (
    LinearOperatorFullMatrix,
)
from aphin.layers import SystemLayer
from aphin.operators import LinearOperatorSkewSym, LinearOperatorSym

import numpy as np
import random
import logging


class LTILayer(SystemLayer):
    """
    Layer for LTI approximation of the time derivative of the latent variable
    z'(t) = A * z(t) + B * u(t)
    with A = J - R, J skew. sym., R sym.
    The system matrix is decomposed into skew-symmetric J and symmetric parts R for conformity with pH layers
    """

    def __init__(
        self,
        r,
        n_u=None,
        n_mu=None,
        regularizer=None,
        layer_sizes=[],
        activation="elu",
        **kwargs,
    ):
        """
        Initialize the LTILayer.

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
        layer_sizes : list, optional
            Sizes of the layers for parameter dependent weights, by default [].
        activation : str, optional
            Activation function to use, by default "elu".
        **kwargs : dict
            Additional arguments for the SystemLayer base class.
        """
        super(LTILayer, self).__init__(r, n_u, n_mu, regularizer, **kwargs)
        self.n_skew = int((r - 1) * r / 2)  # Number of skew symmetric DOF
        self.n_sym = int((r + 1) * r / 2)  # Number of symmetric DOF
        # Layer sizes for parameter dependent weights
        self.layer_sizes = layer_sizes
        self.activation = activation
        if self.n_mu > 0:
            self.init_parameter_dependent_weights()
        else:
            self.init_weights()

    def init_weights(self):
        """
        Initialize trainable variables
        """
        # add trainable variables
        self.dof_J = self.add_weight(
            name="dof_J",
            shape=(self.n_skew,),
            initializer="uniform",
            trainable=True,
            regularizer=self.regularizer,
            dtype=self.dtype_,
        )
        self.dof_R = self.add_weight(
            name="dof_R",
            shape=(self.n_sym,),
            initializer="uniform",
            trainable=True,
            regularizer=self.regularizer,
            dtype=self.dtype_,
        )
        if self.n_u > 0:
            self.dof_B = self.add_weight(
                name="dof_B",
                shape=(self.r, self.n_u),
                initializer="uniform",
                trainable=True,
                regularizer=self.regularizer,
                dtype=self.dtype_,
            )

    @property
    def n_matrices_dofs(self):
        """
        Number of trainable variables for the matrices.

        Returns
        -------
        int
            Number of trainable variables.
        """
        return self.n_skew + self.n_sym + self.r * self.n_u

    @property
    def dof_split(self):
        """
        Split of the internal degrees of freedom into J, R, B.

        Returns
        -------
        tuple
            Tuple containing the split of degrees of freedom.
        """
        return self.n_skew, self.n_sym, self.r * self.n_u, 0, 0

    def init_parameter_dependent_weights(self):
        """
        Initialize trainable variables that depend on parameters (mu).
        """

        input = tf.keras.Input(shape=(self.n_mu,))
        pred = input
        for i, layer_size in enumerate(self.layer_sizes):
            pred = tf.keras.layers.Dense(
                layer_size,
                activation=self.activation,
                # kernel_regularizer=self.regularizer,
                # bias_regularizer=self.regularizer,
                # activity_regularizer=self.regularizer
            )(pred)
        # output layer to weights
        pred = tf.keras.layers.Dense(
            self.n_matrices_dofs,
            activation="linear",
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            # activity_regularizer=self.regularizer
        )(pred)

        self.parameter_dependent_weight_model = tf.keras.Model(
            inputs=input, outputs=pred
        )

    def get_parameter_dependent_weights(self, mu):
        """
        In case of parameter dependent weights, get the weights for the given parameters.

        Parameters
        ----------
        mu : array-like
            Parameters.

        Returns
        -------
        tuple
            Tuple containing dof_J, dof_R, dof_B, and dof_Q.
        """
        # call fully connected layers
        dofs = self.parameter_dependent_weight_model(mu)

        # split into J, R, B (and Q just implemented for PHQ Layer)
        dof_J, dof_R, dof_B, dof_Q, dof_E = tf.split(dofs, self.dof_split, axis=1)
        # split matrices into correct shape
        dof_J = tf.reshape(
            dof_J,
            (
                -1,
                self.n_skew,
            ),
        )
        dof_R = tf.reshape(
            dof_R,
            (
                -1,
                self.n_sym,
            ),
        )
        dof_B = tf.reshape(dof_B, (-1, self.r, self.n_u))
        if dof_Q.shape[1] != 0:
            dof_Q = tf.reshape(dof_Q, (-1, self.n_sym))
        if dof_E.shape[1] != 0:
            dof_E = tf.reshape(dof_E, (-1, self.n_sym))
        return dof_J, dof_R, dof_B, dof_Q, dof_E

    def get_system_matrices(self, mu=None, n_t=None):
        """
        Return matrices in the format (n_sim, r, r).

        Parameters
        ----------
        mu : array-like, optional
            Parameters, by default None.
        n_t : int, optional
            Number of time steps, required if mu is not None, by default None.

        Returns
        -------
        tuple
            Tuple containing matrices J, R, and B.
        """
        if mu is not None:
            self.dof_J, self.dof_R, self.dof_B, _, _ = (
                self.get_parameter_dependent_weights(mu)
            )
            # convert to matrices
            if n_t is None:
                raise ValueError("n_t is required in the parameter-dependent case.")
            # since all matrices are the same over n_t, remove n_t and transform to shape (n_sim,r,r)
            J = np.reshape(self.J.to_dense().numpy(), (-1, n_t, self.r, self.r))[:, 0, :, :]
            R = np.reshape(self.R.to_dense().numpy(), (-1, n_t, self.r, self.r))[:, 0, :, :]
            B = np.reshape(self.B.to_dense().numpy(), (-1, n_t, self.r, self.n_u))[:, 0, :, :] if self.n_u > 0 else None
        else:
            # convert to matrices
            J = np.expand_dims(self.J.to_dense().numpy(), axis=0)
            R = np.expand_dims(self.R.to_dense().numpy(), axis=0)
            if self.n_u > 0:
                B = np.expand_dims(self.B.to_dense().numpy(), axis=0)
            else:
                B = None

        return J, R, B

    @tf.function
    def call(self, z, u=None, mu=None, training=False):
        """
        Evaluate right-hand side of the ODE system z'(t) = f(z, u, mu) for inputs (z, u, mu).

        Parameters
        ----------
        z : array-like
            System states with shape (n_t * n_s, n).
        u : array-like, optional
            System inputs with shape (n_t * n_s, n_u), by default None.
        mu : array-like, optional
            Parameters, by default None.
        training : bool, optional
            Whether the call is in training mode, by default False.

        Returns
        -------
        tf.Tensor
            Approximation of z'(t).
        """

        # update matrices for given parameters
        if self.n_mu > 0:
            self.dof_J, self.dof_R, self.dof_B, _, _ = (
                self.get_parameter_dependent_weights(mu)
            )
        # evaluate PH approximation of the time derivative of the latent variable
        if self.n_u == 0:
            dz_dt = self.J.matvec(z) - self.R.matvec(z)
        else:
            dz_dt = self.J.matvec(z) - self.R.matvec(z) + self.B.matvec(u)
        return dz_dt

    @property
    def J(self):
        """
        Get the skew-symmetric matrix J.

        Returns
        -------
        tf.Tensor
            Skew-symmetric matrix J.
        """

        if not hasattr(self, "dof_J"):
            return tf.zeros([self.r, self.r])
        if self.n_skew == 0:
            return LinearOperatorSym(tf.zeros([self.r, self.r]))
        return LinearOperatorSkewSym(self.dof_J)

    @property
    def R(self):
        """
        Get the symmetric matrix R.

        Returns
        -------
        tf.Tensor
            Symmetric matrix R.
        """

        if not hasattr(self, "dof_R"):
            return tf.zeros([self.r, self.r])
        if self.n_sym == 0:
            return LinearOperatorSym(tf.zeros([self.r, self.r]))
        # if R is only symmetric instead of sym. pos. def., an arbitrary LTI system can be learned
        return LinearOperatorSym(self.dof_R)

    @property
    def B(self):
        """
        Get the (full) input matrix B.

        Returns
        -------
        tf.Tensor
            Input matrix B.
        """

        if self.n_u == 0:
            return tf.zeros([self.r, self.n_u])
        return LinearOperatorFullMatrix(self.dof_B)
