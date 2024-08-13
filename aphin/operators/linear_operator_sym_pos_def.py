"""
TensorFlow linear operators that represent [batch] matrices with special properties
e.g. symmetric, skew-symmetric, symmetric positive semi-definite matrices
based on their degrees of freedom (DOFs)
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import ops
from tensorflow.python.ops.linalg import linear_operator_util
from aphin.operators.operator_utils import _transpose_last2d


class LinearOperatorSymPosDef(tf.linalg.LinearOperatorFullMatrix):
    """
    `LinearOperator` acting like a [batch] square symmetric positive definite matrix.
    Each square symmetric positive definite matrix with shape (N, N) has a number of
    n_dof = N * (N + 1) / 2 degrees of freedom (DOF).

    This operator acts like a [batch] square symmetric positive definite matrix `A` with shape
    `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
    batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
    an square symmetric `N x N` matrix.  This matrix `A` is not materialized but
    only the degrees of freedom (DOF) of each square symmetric positive definite matrix is stored.
    Only for purposes of broadcasting this shape will be relevant.

    `LinearOperatorSymPosDef` is initialized with a (batch) vector that contains
    the DOF of each square symmetric positive definite matrix.
    """

    def __init__(
        self,
        dof,
        is_non_singular=None,
        is_self_adjoint=None,
        is_positive_definite=None,
        is_square=None,
        name="LinearOperatorSymPosDef",
        epsilon=1e-12,
    ):
        """
        Initialize a `LinearOperatorSymPosDef`.

        A symmetric positive definite matrix is a square matrix that is both symmetric
        and positive definite. This class represents a linear operator in the form of a
        symmetric positive definite matrix.

        Parameters
        ----------
        dof : array-like
            Degrees of freedom of a square symmetric positive definite matrix. Expected shape is (n_dof,)
            where n_dof = N * (N + 1) / 2 for a square symmetric positive definite matrix of shape (N, N).
        is_non_singular : bool, optional
            Indicates if the matrix is non-singular. Default is None.
        is_self_adjoint : bool, optional
            Indicates if the matrix is self-adjoint. For symmetric matrices, this should be True as they are trivially self-adjoint. Default is None.
        is_positive_definite : bool, optional
            Indicates if the matrix is positive definite. Default is None.
        is_square : bool, optional
            Indicates if the matrix is square. Must be True for symmetric positive definite matrices. Default is None.
        name : str, optional
            Name of the operator. Default is "LinearOperatorSymPosDef".
        epsilon : float, optional
            Regularization parameter to ensure positive definiteness. Default is 1e-12.
        """

        with ops.name_scope(name, values=[dof]):
            self._dof = linear_operator_util.convert_nonref_to_tensor(dof, name="dof")

            self._chol = tfp.math.fill_triangular(self._dof, upper=False)
            self._chol_t = _transpose_last2d(self._chol)
            # add regularization for Q to become pos. def instead of only semi-def.
            reg_matrix = epsilon * tf.eye(self._chol.shape[-1])
            self._matrix = tf.matmul(self._chol, self._chol_t) + reg_matrix
            # Check and auto-set hints.
            if is_square is False:
                raise ValueError(
                    "Only square skew symmetric operators currently supported."
                )
            is_square = True
            if is_self_adjoint is False:
                raise ValueError("Real symmetric operators are always self adjoint.")
            is_self_adjoint = True

            super(LinearOperatorSymPosDef, self).__init__(
                matrix=self._matrix,
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                is_square=is_square,
                name=name,
            )
