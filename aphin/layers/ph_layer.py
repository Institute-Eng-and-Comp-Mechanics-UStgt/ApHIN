"""
TensorFlow layers for identification of dynamical systems
e.g. LTI systems (LTILayer), port-Hamiltonian systems (PHLayer), ...
intended for standalone use or the use in the latent space of an autoencoder.
All layers approximate the right-hand side of the ODE given states z(t), inputs u(t) and parameters mu
and should be trained using reference values of the states, inputs, parameters and the left-hand side of the ODE
which is assumed to depend only on the time derivative of the states.
"""

import tensorflow as tf
from aphin.layers import LTILayer
from aphin.operators import (
    LinearOperatorSym,
    LinearOperatorSymPosSemiDef,
)


class PHLayer(LTILayer):
    """
    Layer for port-Hamiltonian (pH) approximation of the time derivative of the latent variable
    z'(t) = (J - R) * z(t) + B * u(t)
    with J skew-symmetric and R symmetric positive definite.
    """

    def __init__(self, r, n_u=None, n_mu=None, regularizer=None, **kwargs):
        """
        Initialize the PHLayer.

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
            Additional arguments for the LTILayer base class.
        """
        super(PHLayer, self).__init__(r, n_u, n_mu, regularizer, **kwargs)

    @property
    def R(self):
        """
        Get the symmetric positive definite matrix R.

        Returns
        -------
        tf.Tensor
            Symmetric positive definite matrix R.
        """

        if self.n_sym == 0:
            return LinearOperatorSym(tf.zeros([self.r, self.r]))
        # for a pH system R needs to be symmetric and positive definite
        return LinearOperatorSymPosSemiDef(self.dof_R)
