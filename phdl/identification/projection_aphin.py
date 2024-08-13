import numpy as np
import logging
from .aphin import APHIN
import tensorflow as tf
import tensorflow_probability as tfp

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class ProjectionAPHIN(APHIN):
    """
    Projection-Conserving autoencoder-based port-Hamiltonian Identification Network (ApHIN)
    This is an implementation of an autoencoder that really is a projection, i.e. AE(x) = AE(AE(x)) with AE=Enc(Dec(x))

    Encoder:
        z = h o ... o h o psi^t @ x
            with nonlinear transformation function h( )
    Decoder:
        (phi + (I - phi(psi^T @ phi)^-1 @ psi^T) H) o h^-1 o ... o h^-1 o (psi^t @ phi)^-1 @ z
            with inverse nonlinear transformation function h( )
            and nonlinear Decoder H( )
    """

    def __init__(self, n_transf=0, psi=None, phi=None, **kwargs):
        """
        Initialize the Projection-Conserving Port Hamiltonian Autoencoder.

        Parameters
        ----------
        n_transf : int, optional
            Number of nonlinear invertible transformations applied on the latent variables, by default 0.
        psi : array-like, optional
            Encoder matrix, by default None.
        phi : array-like, optional
            Decoder matrix, by default None.
        kwargs : dict
            Additional arguments for APHIN.
        """

        if not hasattr(self, "config"):
            self._init_to_config(locals())

        self.n_transf = n_transf

        if psi is not None:
            self.psi = tf.Variable(self.psi, name="psi", trainable=False)
        else:
            self.psi = None
        if phi is not None:
            self.phi = tf.Variable(phi, name="phi", trainable=False)
        else:
            self.phi = None

        super(ProjectionAphin, self).__init__(**kwargs)

    def init_weights(self):
        """
        Initialize weights (including projection matrices) for the autoencoder.

        Returns
        -------
        None
        """
        # Initialize phi and psi if they are not predetermined
        if self.phi is None:
            self.phi = self.add_weight(
                name="phi",
                shape=(self.pca_order, self.reduced_order),
                dtype=self.dtype_,
                initializer="orthogonal",
                trainable=True,
            )

        if self.psi is None:
            self.psi = self.add_weight(
                name="psi",
                shape=(self.pca_order, self.reduced_order),
                dtype=self.dtype_,
                initializer="orthogonal",
                trainable=True,
            )

        # weights of nonlinear transformations in latent space
        self.nonlin_transf = []
        for i in range(self.n_transf):
            # # initialize weights and biases
            w_i = self.add_weight(
                name=f"W_{i}",
                shape=(self.reduced_order, self.reduced_order),
                dtype=self.dtype_,
                initializer="glorot_normal",
                trainable=True,
            )
            b_i = self.add_weight(
                name=f"b_{i}",
                shape=(self.reduced_order,),
                dtype=self.dtype_,
                initializer="glorot_normal",
                trainable=True,
            )
            self.nonlin_transf.append([w_i, b_i])

    def build_encoder(self, z_pca):
        """
        Build the encoder part of the autoencoder.
        z = h o ... o h o psi^t @ x with nonlinear transformation function h( )

        Parameters
        ----------
        z_pca : array-like
            Input to the autoencoder.

        Returns
        -------
        z : array-like
            Encoded latent variable.
        """

        z = z_pca

        # Linear projection z = psi^T @ x
        self.encoder_projection = EncoderProjection(self.phi, self.psi)
        z = self.encoder_projection(z)

        # Nonlinear transformations h o ... o h(z) with h(z) = act(W @ z + b)
        for i in range(self.n_transf):
            z = EncoderNonlinearTransformation(
                self.nonlin_transf[i][0],
                self.nonlin_transf[i][1],
                activation_custom,
                dtype=self.dtype_,
            )(z)

        return z

    def build_decoder(self, z):
        """
        Build the decoder part of the autoencoder.
        Decoder: (phi + (I - phi(psi^T @ phi)^-1 @ psi^T) H) o h^-1 o ... o h^-1 o (psi^t @ phi)^-1 @ z
        with invertible nonlinear transformation function h( ) and nonlinear Decoder H( )

        Parameters
        ----------
        z : array-like
            Latent variable.

        Returns
        -------
        z_dec : array-like
            Decoded variable.
        """
        z_ = z

        # Nonlinear transformations h^-1 o ... o h^-1(z) with h^-1(z) = (act^-1(z_1) - b) @ W^-1
        for i in range(self.n_transf - 1, -1, -1):
            z_ = DecoderNonlinearTransformation(
                self.nonlin_transf[i][0],
                self.nonlin_transf[i][1],
                activation_custom_inv,
                dtype=self.dtype_,
            )(z_)

        # Linear latent transformation z = (psi^T @ phi)^-1 @ z
        z_ = DecoderLatentTransformation(self.phi, self.psi)(z_)
        # linear backprojection z_lin = phi @ z_
        z_1 = DecoderLinearProjection(self.phi, self.psi)(z_)

        # decoding to original dimension with nonlinear decoder x_nl = H(z)
        for n_neurons in reversed(self.layer_sizes):
            z_ = tf.keras.layers.Dense(n_neurons, activation=self.activation)(z_)
        z_ = tf.keras.layers.Dense(self.pca_order, activation="linear")(z_)

        # combine linear and nonlinear decoding parts
        z_2 = DecoderNonlinearProjection(self.phi, self.psi, dtype=self.dtype_)(z_)
        z_dec = tf.add(z_1, z_2)

        return z_dec

    def build_nonlinear_autoencoder(self, z_pca):
        """
        Build a projection-conserving autoencoder.

        Parameters
        ----------
        z_pca : array-like
            Input to the autoencoder.

        Returns
        -------
        tuple
            Tuple containing the encoded and decoded variables.
        """
        # Encoder
        z = self.build_encoder(z_pca)
        # Decoder
        z_dec = self.build_decoder(z)

        return z, z_dec


class EncoderProjection(tf.keras.layers.Layer):
    """
    Linear projection/encoding following enc: x to z with z = psi^T @ x
    """

    def __init__(self, phi, psi):
        """
        Initialize the EncoderProjection layer.

        Parameters
        ----------
        phi : array-like
            Projection matrix 1.
        psi : array-like
            Projection matrix 2.
        """
        super(EncoderProjection, self).__init__()
        self.phi = phi
        self.psi = psi

    def build(self, input_shape):
        """
        Build the layer (conformity as weights are shared between many layers and are passed in the init function).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input.

        Returns
        -------
        None
        """
        pass

    def call(self, inputs):
        """
        Perform the linear projection x to z with z = psi^T @ x

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        output : array-like
            Projected data.
        """
        output = inputs @ self.psi
        return output


class DecoderLatentTransformation(tf.keras.layers.Layer):
    """
    Linear Transformation of the form: z to z_ with z_ = (psi^T @ phi)^-1 @ z
    """

    def __init__(self, phi, psi):
        """
        Initialize the DecoderLatentTransformation layer.

        Parameters
        ----------
        phi : array-like
            Projection matrix 1 of shape (n, r)
        psi : array-like
            Projection matrix 2 of shape (n, r)
        """

        super(DecoderLatentTransformation, self).__init__()
        self.phi = phi
        self.psi = psi

    def build(self, input_shape):
        """
        Build the layer (conformity as weights are shared between many layers and are passed in the init function).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input.

        Returns
        -------
        None
        """
        pass

    def call(self, inputs):
        """
        Perform the linear transformation z to z_ with z_ = (psi^T @ phi)^-1 @ z = w @ z.

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        output : array-like
            Transformed data.
        """
        # rewrite equation using tf.linalg.solve
        output = tf.transpose(
            tf.linalg.solve(tf.transpose(self.psi) @ self.phi, tf.transpose(inputs))
        )

        return output


class DecoderLinearProjection(tf.keras.layers.Layer):
    """
    Linear backprojection/decoding following dec_lin: z_ to x_lin with x_lin = phi @ z_
    """

    def __init__(self, phi, psi):
        """
        Initialize the DecoderLinearProjection layer.

        Parameters
        ----------
        phi : array-like
            Projection matrix 1 of shape (n, r).
        psi : array-like
            Projection matrix 2 of shape (n, r).
        """
        super(DecoderLinearProjection, self).__init__()
        self.phi = phi
        self.psi = psi

    def build(self, input_shape):
        """
        Build the layer (conformity as weights are shared between many layers and are passed in the init function).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input.

        Returns
        -------
        None
        """
        pass

    def call(self, inputs):
        """
        Perform the linear backprojection z_ to x_lin with x_lin = phi @ z_

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        output : array-like
            Backprojected data.
        """
        output = inputs @ tf.transpose(self.phi)
        return output


class DecoderNonlinearProjection(tf.keras.layers.Layer):
    """
    Nonlinear/decoding following dec_nl: z_ to x_nl with x_nl = (I - phi @ (psi^T @ phi)^-1 @ psi^T) @ H(z_)
    """

    def __init__(self, phi, psi, dtype=tf.float32):
        """
        Initialize the DecoderNonlinearProjection layer.

        Parameters
        ----------
        phi : array-like
            Projection matrix 1 of shape (n, r).
        psi : array-like
            Projection matrix 2 of shape (n, r).
        dtype : tf.DType, optional
            Data type, by default tf.float32.
        """
        super(DecoderNonlinearProjection, self).__init__()
        self.dtype_ = dtype
        self.phi = phi
        self.psi = psi

    def build(self, input_shape):
        """
        Build the layer (conformity as weights are shared between many layers and are passed in the init function).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input.

        Returns
        -------
        None
        """
        pass

    def call(self, inputs):
        """
        Perform the nonlinear projection z_ to x_nl with x_nl = (I - phi @ (psi^T @ phi)^-1 @ psi^T) @ H(z_).

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        output : array-like
            Nonlinear projected data.
        """
        # identity matrix
        I = tf.eye(tf.shape(self.phi)[0], dtype=self.dtype_)
        # reformulate using tf.linalg.solve
        W = I - self.phi @ (
            tf.linalg.solve(tf.transpose(self.psi) @ self.phi, tf.transpose(self.psi))
        )
        output = inputs @ tf.transpose(W)

        return output


class EncoderNonlinearTransformation(tf.keras.layers.Layer):
    """
    Nonlinear transformation in the latent space on the encoder side following: z to z_ with z_ = act(W @ z + b)
    """

    def __init__(self, W, b, activation, dtype=tf.float32):
        """
        Initialize the EncoderNonlinearTransformation layer.

        Parameters
        ----------
        W : array-like
            Weighting matrix of shape (r, r).
        b : array-like
            Bias of shape (r, 1).
        activation : callable
            Activation function.
        dtype : tf.DType, optional
            Data type, by default tf.float32.
        """

        super(EncoderNonlinearTransformation, self).__init__()
        self.dtype_ = dtype
        self.W = W
        self.b = b
        self.activation = activation

    def build(self, input_shape):
        """
        Build the layer (conformity as weights are shared between many layers and are passed in the init function).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input.

        Returns
        -------
        None
        """
        pass

    def call(self, inputs):
        """
        Perform the nonlinear transformation z to z_ with z_ = act(W @ z + b).

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        output : array-like
            Transformed data.
        """
        output = self.activation(
            inputs @ tf.transpose(self.W) + self.b, dtype=self.dtype_
        )
        return output


class DecoderNonlinearTransformation(tf.keras.layers.Layer):
    """
    Nonlinear transformation in the latent space on the decoder side (needs to be inverse of the latent correspondent):
    z_1 to z_2 with z_2 = (act^-1(z_1) - b) @ W^-1
    """

    def __init__(self, W, b, activation, dtype=tf.float32):
        """
        Initialize the DecoderNonlinearTransformation layer.

        Parameters
        ----------
        W : array-like
            Weighting matrix (r, r).
        b : array-like
            Bias (r, 1).
        activation : callable
            Inverse activation function.
        dtype : tf.DType, optional
            Data type, by default tf.float32.
        """
        super(DecoderNonlinearTransformation, self).__init__()
        self.dtype_ = dtype
        self.W = W
        self.b = b
        self.activation = activation

    def build(self, input_shape):
        """
        Build the layer (conformity as weights are shared between many layers and are passed in the init function).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input.

        Returns
        -------
        None
        """
        pass

    def call(self, inputs):
        """
        Perform the inverse nonlinear transformation z_1 to z_2 with z_2 = (act^-1(z_1) - b) @ W^-1.

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        output : array-like
            Transformed data.
        """

        # rewrite equation using tf.linalg.solve
        output = tf.transpose(
            tf.linalg.solve(
                self.W,
                tf.transpose(self.activation(inputs, dtype=self.dtype_) - self.b),
            )
        )
        return output


def cosec(x):
    """
    Calculate the cosecant of x: 1/sin(x).

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    array-like
        Cosecant of the input.
    """
    return tf.math.reciprocal(tf.sin(x))


def sec(x):
    """
    Calculate the secant of x: 1/cos(x).

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    array-like
        Secant of the input.

    """
    return tf.math.reciprocal(tf.cos(x))


def activation_custom(x, alpha=np.pi / 8, dtype=tf.float32):
    """
    Invertible nonlinear activation function.
    see Otto, S. E. (2022). Advances in Data-Driven Modeling and Sensing for High-Dimensional Nonlinear Systems
    Doctoral dissertation, Princeton University. Eq. 3.71

    Parameters
    ----------
    x : array-like
        Input data.
    alpha : float, optional
        Parameter for the activation function, by default np.pi / 8.
    dtype : tf.DType, optional
        Data type, by default tf.float32.

    Returns
    -------
    array-like
        Transformed data.
    """
    alpha = tf.constant(alpha, dtype=dtype)
    a = tf.square(cosec(alpha)) - tf.square(sec(alpha))
    b = tf.square(cosec(alpha)) + tf.square(sec(alpha))
    return (b * x + tf.sqrt((b**2 - a**2) * tf.square(x) + 2 * a)) / a


def activation_custom_inv(x, alpha=np.pi / 8, dtype=tf.float32):
    """
    Inverse of the nonlinear invertible activation function.
    see Otto, S. E. (2022). Advances in Data-Driven Modeling and Sensing for High-Dimensional Nonlinear Systems
    Doctoral dissertation, Princeton University. Eq. 3.71

    Parameters
    ----------
    x : array-like
        Input data.
    alpha : float, optional
        Parameter for the activation function, by default np.pi / 8.
    dtype : tf.DType, optional
        Data type, by default tf.float32.

    Returns
    -------
    array-like
        Transformed data.
    """
    alpha = tf.constant(alpha, dtype=dtype)
    a = tf.square(cosec(alpha)) - tf.square(sec(alpha))
    b = tf.square(cosec(alpha)) + tf.square(sec(alpha))
    return (b * x - tf.sqrt((b**2 - a**2) * tf.square(x) + 2 * a)) / a
