"""
TensorFlow layers for identification of dynamical systems
e.g. LTI systems (LTILayer), port-Hamiltonian systems (PHLayer), ...
intended for standalone use or the use in the latent space of an autoencoder.
All layers approximate the right-hand side of the ODE given states z(t), inputs u(t) and parameters mu
and should be trained using reference values of the states, inputs, parameters and the left-hand side of the ODE
which is assumed to depend only on the time derivative of the states.
"""

import tensorflow as tf
import numpy as np

from aphin.operators import LinearOperatorSym, LinearOperatorSymPosSemiDef
from aphin.layers import PHLayer, PHQLayer


class DescriptorPHLayer(PHLayer):
    """
    Layer for port-Hamiltonian (pH) approximation with a descriptive matrix of the time derivative of the latent variable
    E * z'(t) = (J - R) * z(t) + B * u(t)
    with J skew-symmetric, R symmetric positive definite, and E symmetric.
    """

    def __init__(self, r, n_u=None, n_mu=None, regularizer=None, **kwargs):
        """
        Initialize the DescriptorPHLayer.

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
        super(DescriptorPHLayer, self).__init__(r, n_u, n_mu, regularizer, **kwargs)

    def init_weights(self):
        """
        Initialize trainable variables
        """
        # add trainable variables
        super(DescriptorPHLayer, self).init_weights()
        self.dof_E = self.add_weight(
            name="dof_E",
            shape=(self.n_sym,),
            initializer="uniform",
            trainable=True,
            dtype=self.dtype_,
        )

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
            Tuple containing matrices J, R, B, and E.
        """
        J, R, B = super().get_system_matrices(mu, n_t)
        if mu is not None:
            if n_t is None:
                raise ValueError("n_t is required in the parameter-dependent case.")
            _, _, _, _, self.dof_E = self.get_parameter_dependent_weights(mu)
            E = np.reshape(self.E.to_dense().numpy(), (-1, n_t, self.r, self.r))
            E = E[:, 0, :, :]
        else:
            # convert to matrices
            E = np.expand_dims(self.E.to_dense().numpy(), axis=0)

        return J, R, B, E

    @property
    def dof_split(self):
        """
        Split of the internal degrees of freedom into J, R, B, Q, and E.

        Returns
        -------
        tuple
            Tuple containing the split of degrees of freedom.
        """
        return self.n_skew, self.n_sym, self.r * self.n_u, 0, self.n_sym

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

        return LinearOperatorSymPosSemiDef(self.dof_E)


class DescriptorPHQLayer(PHQLayer):
    """
    Layer for port-Hamiltonian (pH) approximation with a descriptive matrix of the time derivative of the latent variable
    E * z'(t) = (J - R) * Q * z(t) + B * u(t)
    with J skew-symmetric, R symmetric positive definite, Q symmetric positive definite, and E symmetric.
    This is a generalization of DescriptorPHLayer.
    """

    def __init__(self, r, n_u=None, n_mu=None, regularizer=None, **kwargs):
        """
        Initialize the DescriptorPHQLayer.

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
            Additional arguments for the PHQLayer base class.
        """
        super(DescriptorPHQLayer, self).__init__(r, n_u, n_mu, regularizer, **kwargs)

    def init_weights(self):
        """
        Initialize trainable variables
        """
        # add trainable variables
        super(DescriptorPHQLayer, self).init_weights()
        self.dof_E = self.add_weight(
            name="dof_E",
            shape=(self.n_sym,),
            initializer="uniform",
            trainable=True,
            dtype=self.dtype_,
        )

    @property
    def dof_split(self):
        """
        Split of the internal degrees of freedom into J, R, B, Q, and E.

        Returns
        -------
        tuple
            Tuple containing the split of degrees of freedom.
        """
        return self.n_skew, self.n_sym, self.r * self.n_u, self.n_sym, self.n_sym

    @property
    def n_matrices_dofs(self):
        """
        Number of trainable variables for the matrices.

        Returns
        -------
        int
            Number of trainable variables.
        """
        return self.n_skew + self.n_sym + self.r * self.n_u + self.n_sym + self.n_sym

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
            Tuple containing matrices J, R, B, Q, and E.
        """
        J, R, B, Q = super().get_system_matrices(mu, n_t)
        if mu is not None:
            if n_t is None:
                raise ValueError("n_t is required in the parameter-dependent case.")
            _, _, _, _, self.dof_E = self.get_parameter_dependent_weights(mu)
            E = np.reshape(self.E.to_dense().numpy(), (-1, n_t, self.r, self.r))
            E = E[:, 0, :, :]
        else:
            # convert to matrices
            E = np.expand_dims(self.E.to_dense().numpy(), axis=0)

        return J, R, B, Q, E

    @property
    def E(self):
        """
        Get the symmetric matrix E.

        Returns
        -------
        tf.Tensor
            Symmetric matrix E.
        """
        raise NotImplementedError("pH condition not fullfilled.")
        return LinearOperatorSym(self.dof_E)
