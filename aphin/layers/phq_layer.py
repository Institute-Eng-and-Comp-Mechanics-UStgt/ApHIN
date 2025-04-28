"""
TensorFlow layers for identification of dynamical systems
e.g. LTI systems (LTILayer), port-Hamiltonian systems (PHLayer), ...
intended for standalone use or the use in the latent space of an autoencoder.
All layers approximate the right-hand side of the ODE given states z(t), inputs u(t) and parameters mu
and should be trained using reference values of the states, inputs, parameters and the left-hand side of the ODE
which is assumed to depend only on the time derivative of the states.
"""

import logging
import random
import tensorflow as tf
import numpy as np

from aphin.operators import LinearOperatorSymPosDef, LinearOperatorSym
from aphin.layers import PHLayer


class PHQLayer(PHLayer):
    """
    Layer for port-Hamiltonian (pH) approximation of the time derivative of the latent variable
    z'(t) = (J - R) * Q * z(t) + B * u(t)
    with J skew-symmetric, R symmetric positive definite, and Q symmetric positive definite.
    This is a generalization of PHLayer.
    """

    def __init__(self, r, n_u=None, n_mu=None, regularizer=None, **kwargs):
        """
        Initialize the PHQLayer.

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
        **kwargs : dict
            Additional arguments for the PHLayer base class.
        """

        super(PHQLayer, self).__init__(r, n_u, n_mu, regularizer, **kwargs)

        # regularization parameter to ensure positive definiteness in case self.n_sym > 0
        if self.dtype_ is tf.float32:
            self.epsilon = 1e-6
        else:
            self.epsilon = 1e-12

    def init_weights(self):
        """
        Initialize trainable variables
        """
        # add trainable variables
        super(PHQLayer, self).init_weights()
        self.dof_Q = self.add_weight(
            name="dof_Q",
            shape=(self.n_sym,),
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
        return self.n_skew + self.n_sym + self.r * self.n_u + self.n_sym

    @property
    def dof_split(self):
        """
        Split of the internal degrees of freedom into J, R, B, and Q.

        Returns
        -------
        tuple
            Tuple containing the split of degrees of freedom.
        """
        return self.n_skew, self.n_sym, self.r * self.n_u, self.n_sym, 0

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
            Tuple containing matrices J, R, B, and Q.
        """
        J, R, B = super().get_system_matrices(mu, n_t)
        if mu is not None:
            if n_t is None:
                raise ValueError("n_t is required in the parameter-dependent case.")
            _, _, _, self.dof_Q, _ = self.get_parameter_dependent_weights(mu)
            Q = np.reshape(self.Q.to_dense().numpy(), (-1, n_t, self.r, self.r))
            Q = Q[:, 0, :, :]
        else:
            # convert to matrices
            Q = np.expand_dims(self.Q.to_dense().numpy(), axis=0)

        return J, R, B, Q

    @tf.function
    def call(self, z, u, mu=None, training=False):
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
            self.dof_J, self.dof_R, self.dof_B, self.dof_Q, self.dof_E = (
                self.get_parameter_dependent_weights(mu)
            )
        # z, u, mu = self.split_inputs(inputs, input_shapes)
        # evaluate PH approximation of the time derivative of the latent variable
        # z, u = tf.cast(z, dtype=self.dtype_), tf.cast(u, dtype=self.dtype_)
        Qz = self.Q.matvec(z)
        if self.r > 1:
            dz_dt = self.J.matvec(Qz) - self.R.matvec(Qz)
        else:
            dz_dt = -self.R.matvec(Qz)
        if self.n_u > 0:
            u = tf.cast(u, dtype=self.dtype_)
            dz_dt += self.B.matvec(u)
        return dz_dt

    @property
    def Q(self):
        """
        Get the symmetric positive definite matrix Q.

        Returns
        -------
        tf.Tensor
            Symmetric positive definite matrix Q.
        """

        if self.n_sym == 0:
            return LinearOperatorSym(tf.zeros([self.r, self.r]))
        # for a pH system Q needs to be symmetric and positive definite
        return LinearOperatorSymPosDef(self.dof_Q, epsilon=self.epsilon)
