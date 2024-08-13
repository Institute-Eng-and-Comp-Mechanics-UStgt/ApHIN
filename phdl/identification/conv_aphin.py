import numpy as np
import logging
from .aphin import APHIN
import tensorflow as tf

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class ConvAPHIN(APHIN):
    """
    Convolutional autoencoder-based port-Hamiltonian Identification Network (Conv-ApHIN).
    Model to discover low-dimensional dynamics of a high-dimensional system using a convolutional autoencoder and pHIN
    """

    def __init__(
        self,
        reduced_order,
        n_filters,
        kernel_size,
        strides,
        system_layer=None,
        **kwargs,
    ):
        """
        Initialize the ConvAPHIN model.
        A convolution has the output shape
        Encoder:  4+D tensor with shape: `batch_shape + (filters, new_rows, new_cols)
        Decoder: new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
        new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1])

        Parameters
        ----------
        reduced_order : int
            The reduced order of the system.
        n_filters : list of int
            Number of filters for each convolutional layer.
        kernel_size : list of int
            Size of the kernel for each convolutional layer.
        strides : list of int
            Strides for each convolutional layer.
        system_layer : tf.keras.layers.Layer, optional
            The system layer, by default None.
        **kwargs : dict
            Additional arguments for the APHIN base class.
        """
        if "use_pca" in kwargs.keys():
            assert (
                kwargs["use_pca"] != True
            ), "PCA cannot be used with convolutinoal autoencoder"
        if "pca_scaling" in kwargs.keys():
            logging.warn("pca_scaling has no effect for ConvAPHIN. No scaling applied.")
        if not hasattr(self, "config"):
            self._init_to_config(locals())
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        super(ConvAPHIN, self).__init__(reduced_order, **kwargs)

    def build_autoencoder(self, x):
        """
        Build the encoder and decoder of the autoencoder.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        tuple
            Tuple containing the input tensor, dummy PCA tensor, encoded tensor, decoded tensor, and reconstructed tensor.
        """
        x_input = tf.keras.Input(shape=(x.shape[1], x.shape[2]))
        z_pca_dummy = 1 * x_input
        # second part of the encoder and first part of the decoder is a nonlinear part
        z, z_dec = self.build_nonlinear_autoencoder(z_pca_dummy)
        x_rec = 1 * z_dec
        return x_input, z_pca_dummy, z, z_dec, x_rec

    def build_nonlinear_autoencoder(self, x_input):
        """
        Build the convolutional autoencoder with specified layers and filter sizes.

        Parameters
        ----------
        x_input : tf.Tensor
            Input tensor to the encoder.

        Returns
        -------
        tuple
            Tuple containing the encoded tensor and the decoded tensor.
        """

        # in case the data only consists of one channel and not multiple ones create axis for this
        if len(x_input.shape) == 3:
            x_input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(
                x_input
            )
        input_shape = x_input.shape

        z_ = x_input
        for n_filters, kernel_size, strides in zip(
            self.n_filters, self.kernel_size, self.strides
        ):
            z_ = tf.keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                activation=self.activation,
                activity_regularizer=self.regularizer,
            )(z_)
            # pooling layer
            # z_ = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(z_)
            # print(z_.shape)
        z_shape = z_.shape
        z_ = tf.keras.layers.Flatten()(z_)
        z_shape2 = z_.shape
        z = tf.keras.layers.Dense(
            self.reduced_order,
            activation="linear",
            # activity_regularizer=self.regularizer
        )(z_)

        # decoder part
        z_dec = tf.keras.layers.Dense(
            z_shape2[1],
            activation=self.activation,
            activity_regularizer=self.regularizer,
        )(z)
        z_dec = tf.keras.layers.Reshape(z_shape[1:])(z_dec)

        # deconvolutional part
        for n_filters, kernel_size, strides in zip(
            self.n_filters[-2::-1], self.kernel_size[-2::-1], self.strides[-2::-1]
        ):
            # unpooling layer
            # z_dec = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(z_dec)
            z_dec = tf.keras.layers.Conv2DTranspose(
                filters=n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                activation=self.activation,
                activity_regularizer=self.regularizer,
            )(z_dec)

        z_dec = tf.keras.layers.Conv2DTranspose(
            filters=input_shape[-1],
            kernel_size=self.kernel_size[0],
            strides=self.strides[0],
            padding="same",
            activation="linear",
            # activity_regularizer=self.regularizer
        )(z_dec)
        # reshape to original shape
        # z_dec = tf.keras.layers.Flatten()(z_dec)
        return z, z_dec

    def get_loss_second_part(self, xr, dz_dxr, dxr_dt, z, u, mu):
        """
        Calculate the second part of the loss function.
        In contrast to the classic APHIN, our data (and its time derivative) are multidimensional.
        Consequently, we need to vectorize the data before we can calculate the loss.

        Parameters
        ----------
        xr : array-like
            Reconstructed data.
        dz_dxr : array-like
            Derivative of the latent variables with respect to the reconstructed data.
        dxr_dt : array-like
            Time derivative of the reconstructed data.
        z : array-like
            Latent variables.
        u : array-like
            Control inputs.
        mu : array-like
            Parameters.

        Returns
        -------
        tf.Tensor
            The second part of the loss.
        """

        # reshape data from (n_samples, n_pixels, n_pixels, n_channels) to (n_samples, n_pixels, n_pixels, n_channels)
        dz_dxr, dxr_dt = self.reshape_conv_data(dz_dxr, dxr_dt)

        return super(ConvAPHIN, self).get_loss_second_part(xr, dz_dxr, dxr_dt, z, u, mu)

    def reshape_conv_data(self, dz_dxr, dxr_dt):
        """
        In contrast to the classic APHIN, our data (and its time derivative) are multidimensional.
        Consequently, we need to vectorize the data before we can calculate the loss.

        Parameters
        ----------
        dz_dxr : array-like
            Derivative of the latent variables with respect to the reconstructed data.
        dxr_dt : array-like
            Time derivative of the reconstructed data.

        Returns
        -------
        tuple
            Tuple containing the reshaped derivatives and time derivatives.
        """
        dz_dxr = tf.reshape(
            dz_dxr, (-1, dz_dxr.shape[1], dz_dxr.shape[2] * dz_dxr.shape[3])
        )
        dxr_dt = tf.reshape(
            dxr_dt, (-1, dxr_dt.shape[1] * dxr_dt.shape[2], dxr_dt.shape[3])
        )
        return dz_dxr, dxr_dt

    def reshape_dxr_dz(self, dxr_dz):
        """
        Reshape the derivative of the reconstructed data with respect to the latent variables.

        Parameters
        ----------
        dxr_dz : array-like
            Derivative of the reconstructed data with respect to the latent variables.

        Returns
        -------
        tf.Tensor
            Reshaped derivative tensor.
        """
        return tf.reshape(
            dxr_dz, (-1, dxr_dz.shape[1] * dxr_dz.shape[2], dxr_dz.shape[4])
        )
