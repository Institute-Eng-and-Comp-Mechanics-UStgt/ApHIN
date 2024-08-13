"""
TensorFlow layers for identification of dynamical systems
e.g. LTI systems (LTILayer), port-Hamiltonian systems (PHLayer), ...
intended for standalone use or the use in the latent space of an autoencoder.
All layers approximate the right-hand side of the ODE given states z(t), inputs u(t) and parameters mu
and should be trained using reference values of the states, inputs, parameters and the left-hand side of the ODE
which is assumed to depend only on the time derivative of the states.
"""

import tensorflow as tf

from aphin.operators import LinearOperatorSym
from aphin.layers import PHLayer, PHQLayer


class DescriptorPHLayer(PHLayer):
    """
    Layer for port-Hamiltonian (pH) approximation with a descriptive matrix of the time derivative of the latent variable
    E * z'(t) = (J - R) * z(t) + B * u(t)
    with J skew-symmetric, R symmetric positive definite, and E symmetric.
    """

    def __init__(self, r, n_u=None, regularizer=None, **kwargs):
        """
        Initialize the DescriptorPHLayer.

        Parameters
        ----------
        r : int
            Number of latent variables.
        n_u : int, optional
            Number of inputs, by default None.
        regularizer : tf.keras.regularizers.Regularizer, optional
            Regularizer object for trainable variables of the class, by default None.
        **kwargs : dict
            Additional arguments for the PHLayer base class.
        """
        super(DescriptorPHLayer, self).__init__(r, n_u, regularizer, **kwargs)
        self.dof_E = self.add_weight(
            name="dof_E",
            shape=(self.n_sym,),
            initializer="uniform",
            trainable=True,
            dtype=self.dtype_,
        )

    @tf.function
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
        return self.E.matvec(dz_dt)

    @property
    def E(self):
        """
        Get the symmetric matrix E.

        Returns
        -------
        tf.Tensor
            Symmetric matrix E.
        """

        return LinearOperatorSym(self.dof_E)


class DescriptorPHQLayer(PHQLayer):
    """
    Layer for port-Hamiltonian (pH) approximation with a descriptive matrix of the time derivative of the latent variable
    E * z'(t) = (J - R) * Q * z(t) + B * u(t)
    with J skew-symmetric, R symmetric positive definite, Q symmetric positive definite, and E symmetric.
    This is a generalization of DescriptorPHLayer.
    """

    def __init__(self, r, n_u=None, regularizer=None, **kwargs):
        """
        Initialize the DescriptorPHQLayer.

        Parameters
        ----------
        r : int
            Number of latent variables.
        n_u : int, optional
            Number of inputs, by default None.
        regularizer : tf.keras.regularizers.Regularizer, optional
            Regularizer object for trainable variables of the class, by default None.
        **kwargs : dict
            Additional arguments for the PHQLayer base class.
        """
        super(DescriptorPHQLayer, self).__init__(r, n_u, regularizer, **kwargs)
        self.dof_E = self.add_weight(
            name="dof_E",
            shape=(self.n_sym,),
            initializer="uniform",
            trainable=True,
            dtype=self.dtype_,
        )

    @tf.function
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
        return self.E.matvec(dz_dt)

    @property
    def E(self):
        """
        Get the symmetric matrix E.

        Returns
        -------
        tf.Tensor
            Symmetric matrix E.
        """
        return LinearOperatorSym(self.dof_E)
