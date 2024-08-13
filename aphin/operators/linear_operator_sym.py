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


class LinearOperatorSym(tf.linalg.LinearOperatorFullMatrix):
    """
    `LinearOperator` acting like a [batch] square symmetric matrix.
    Each square symmetric matrix with shape (N, N) has a number of
    n_dof = N * (N + 1) / 2 degrees of freedom (DOF).

    This operator acts like a [batch] square symmetric matrix `A` with shape
    `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
    batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
    an square symmetric `N x N` matrix.  This matrix `A` is not materialized but
    only the degrees of freedom (DOF) of each square symmetric matrix are stored.
    Only for purposes of broadcasting this shape will be relevant.

    `LinearOperatorSym` is initialized with a (batch) vector that contains
    the DOF of each square symmetric matrix.
    """

    def __init__(
        self,
        dof,
        is_non_singular=None,
        is_self_adjoint=None,
        is_positive_definite=None,
        is_square=None,
        name="LinearOperatorSym",
    ):
        """
        Initialize a `LinearOperatorSym`.

        This class represents a linear operator in the form of a symmetric matrix.

        Parameters
        ----------
        dof : array-like
            Degrees of freedom of a square symmetric matrix. Expected shape is (n_dof,)
            where n_dof = N * (N + 1) / 2 for a square symmetric matrix of shape (N, N).
        is_non_singular : bool, optional
            Indicates if the matrix is non-singular. Default is None.
        is_self_adjoint : bool, optional
            Indicates if the matrix is self-adjoint. For symmetric matrices, this should be True as they are trivially self-adjoint. Default is None.
        is_positive_definite : bool, optional
            Indicates if the matrix is positive definite. Default is None.
        is_square : bool, optional
            Indicates if the matrix is square. Must be True for symmetric matrices. Default is None.
        name : str, optional
            Name of the operator. Default is "LinearOperatorSym".
        """

        with ops.name_scope(name, values=[dof]):
            self._dof = linear_operator_util.convert_nonref_to_tensor(dof, name="dof")

            self._tril = tfp.math.fill_triangular(self._dof, upper=False)
            self._triu = _transpose_last2d(self._tril)
            self._matrix = self._tril + self._triu

            # Check and auto-set hints.
            if is_square is False:
                raise ValueError("Only square symmetric operators currently supported.")
            is_square = True

            super(LinearOperatorSym, self).__init__(
                matrix=self._matrix,
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                is_square=is_square,
                name=name,
            )
