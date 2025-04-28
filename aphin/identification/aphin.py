import numpy as np
from abc import ABC
import logging
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
import matplotlib.pyplot as plt

# own modules
from . import PHBasemodel
from aphin.layers import PHLayer
from aphin.utils import integrators

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class APHIN(PHBasemodel):
    """
    Autoencoder-based port-Hamiltonian Identification Network (ApHIN)
    """

    def __init__(
        self,
        reduced_order,
        pca_order=None,
        x=None,
        u=None,
        mu=None,
        system_layer=None,
        layer_sizes=None,
        activation="selu",
        pca_scaling=False,
        use_pca=True,
        pca_only=False,
        l_rec: float = 1,
        l_dz: float = 1,
        l_dx: float = 1,
        l1=0,
        l2=0,
        dtype=tf.float32,
        **kwargs,
    ):
        """
        Model to discover low-dimensional dynamics of a high-dimensional system using autoencoders and
        a layer for identification of other dynamical systems (see SystemLayer), e.g., a PHLayer (port-Hamiltonian)

        Parameters
        ----------
        reduced_order : int
            Order of the reduced model.
        pca_order : int, optional
            Order of the PCA model.
        x : array-like, optional
            Input data (full states) with shape (n_sim*n_t, n_f) with n_sim simulation scenarios, n_t time steps, n_f features.
        u : array-like, optional
            System inputs with shape (n_sim*n_t, n_u) with n_sim simulation scenarios, n_t time steps, n_u inputs.
        mu : array-like, optional
            Parameter data (n_sim*n_t, n_mu).
        system_layer : SystemLayer or subclass instance, optional
            Instance that learns the reduced system in the latent space.
        layer_sizes : list of int, optional
            Layers of the dense neural network of the non-linear autoencoder part.
        activation : str, optional
            Activation function of the dense neural network of the non-linear autoencoder part.
        pca_scaling : bool, optional
            If PCA modes should be scaled according to singular values.
        use_pca : bool, optional
            If PCA (linear MOR) should be performed before/after the non-linear autoencoder part.
        pca_only : bool, optional
            If only PCA (linear MOR) should be performed instead of the non-linear autoencoder part.
        l_rec : float, optional
            Weight of the reconstruction loss.
        l_dz : float, optional
            Weight of the derivative loss in the latent space.
        l_dx : float, optional
            Weight of the derivative loss in the physical space.
        l1 : float, optional
            L1 regularization factor.
        l2 : float, optional
            L2 regularization factor.
        dtype : str, optional
            Data type for the model.
        kwargs : dict
            Additional keyword arguments.
        """
        # tf.keras.backend.set_floatx(dtype)
        self.dtype_ = dtype
        super(APHIN, self).__init__(dtype=dtype, **kwargs)

        # general parameters
        self.system_optimizer = None
        if layer_sizes is None:
            self.layer_sizes = [10, 10, 10]
        else:
            self.layer_sizes = layer_sizes
        self.activation = tf.keras.activations.get(activation)
        # pca related parameters
        if pca_order is None:
            pca_order = 10 * reduced_order
        self.reduced_order = reduced_order
        self.pca_order = pca_order
        self.pca_scaling_individual = pca_scaling
        self.scale_factor = None
        self.use_pca = use_pca
        self.pca_only = pca_only
        if self.pca_only:
            self.use_pca = True
            if pca_order is not None:
                logging.info(
                    f"pca_only is chosen. Setting pca_order to reduced_order of size {self.reduced_order}."
                )
            self.pca_order = self.reduced_order
            self.layer_sizes = []
        if system_layer is None:
            self.system_layer = PHLayer(self.reduced_order)
        else:
            self.system_layer = system_layer
        # weighting of the different losses
        self.l_rec, self.l_dz, self.l_dx = l_rec, l_dz, l_dx
        self.regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

        # create the model
        x = tf.cast(x, dtype=self.dtype_)
        self.x_shape = x.shape[1:]
        if u is not None:
            u = tf.cast(u, dtype=self.dtype_)
            self.u_shape = u.shape[1:]
        if mu is not None:
            mu = tf.cast(mu, dtype=self.dtype_)
            self.mu_shape = mu.shape[1:]

        if not use_pca:
            self.pca_order = x.shape[1]
        # some subclasses initialize weights before building the model
        if hasattr(self, "init_weights"):
            self.init_weights()
        self.build_model(x, u, mu)

        # create loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.dz_loss_tracker = tf.keras.metrics.Mean(name="dz_loss")
        self.dx_loss_tracker = tf.keras.metrics.Mean(name="dx_loss")
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")

        # decide which loss needs to be evaluated
        # only perform reconstruction if no identification loss is used
        if self.l_dx == 0.0 and self.l_dz == 0.0:
            self.get_loss = self._get_loss_rec
        # calculate loss for first order systems
        else:
            self.get_loss = self._get_loss

        # save parsed arguments
        if not hasattr(self, "config"):
            self._init_to_config(locals())

    def get_trainable_weights(self):
        """
        Returns the trainable weights of the model.

        Returns
        -------
        list
            List of trainable weights.
        """
        return (
            self.encoder.trainable_weights
            + self.decoder.trainable_weights
            + self.system_network.trainable_weights
        )

    def build_model(self, x, u, mu):
        """
        Build the model.

        Parameters
        ----------
        x : array-like
            Full state with shape (n_samples, n_features).
        u : array-like, optional
            Inputs with shape (n_samples, n_inputs).
        mu : array-like, optional
            Parameters with shape (n_samples, n_params).
        """

        x_input, z_pca, z, z_dec, x = self.build_autoencoder(x)

        # System inputs
        if u is not None:
            u_input = tf.keras.Input(shape=(u.shape[1],), dtype=u.dtype)
        else:
            u_input = tf.keras.Input(shape=(0,))

        # Simulation parameters
        if mu is not None:
            mu_input = tf.keras.Input(shape=(mu.shape[1],))
        else:
            mu_input = tf.keras.Input(shape=(0,))

        dz_dt_system = self.system_layer(z, u_input, mu_input)

        # linear autoencoder part from full to intermediate latent space
        self.pca_encoder = tf.keras.Model(
            inputs=x_input, outputs=z_pca, name="pca_encoder"
        )
        self.pca_decoder = tf.keras.Model(inputs=z_dec, outputs=x, name="pca_decoder")
        # nonlinear autoencoder part from intermediate latent space to latent space
        self.nonlinear_encoder = tf.keras.Model(
            inputs=z_pca, outputs=z, name="nonlinear_encoder"
        )
        self.nonlinear_decoder = tf.keras.Model(
            inputs=z, outputs=z_dec, name="nonlinear_decoder"
        )
        # global autoencoder
        self.encoder = tf.keras.Model(inputs=x_input, outputs=z, name="encoder")
        self.decoder = tf.keras.Model(inputs=z, outputs=x, name="decoder")
        # network for discovery of dynamic system (e.g. pH system)
        self.system_network = tf.keras.Model(
            inputs=[z, u_input, mu_input], outputs=dz_dt_system, name="system"
        )

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
            Tuple containing inputs and outputs of the autoencoder.
        """
        x_input = tf.keras.Input(shape=(x.shape[1],), dtype=self.dtype_)
        # first part of the encoder may consist of a linear projection based on PCA
        z_pca = self.build_pca_encoder(x, x_input)
        # second part of the encoder and first part of the decoder is a nonlinear part
        z, z_dec = self.build_nonlinear_autoencoder(z_pca)
        # second part of the decoder is a back projection based on PCA
        x = self.build_pca_decoder(z_dec)

        return x_input, z_pca, z, z_dec, x

    def build_pca_encoder(self, x, x_input):
        """
        Calculate the PCA of the data and build a linear encoder which is equivalent to the PCA.

        Parameters
        ----------
        x : array-like
            Input data.
        x_input : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Encoded PCA tensor.
        """
        # if PCA is used, calculate the PCA and build a linear encoder
        if self.use_pca:
            # calculate the PCA
            pca = TruncatedSVD(n_components=self.pca_order)
            pca.fit(x)
            # use the projection matrix as linear encoder
            self.down = tf.cast(pca.components_, dtype=self.dtype_)
            self.up = tf.cast(pca.components_.T, dtype=self.dtype_)
            self.singular_values = pca.singular_values_
            z_pca = x_input @ tf.transpose(self.down)
            # compute relative reconstruction error
            x_rec = pca.inverse_transform(pca.transform(x))
            x_rel = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
            logging.info(f"Relative reconstruction error of PCA: {x_rel:.4g}")
        # in case no PCA is used, the encoder is just the identity
        else:
            z_pca = x_input * 1

        # if individual scaling is used, scale every feature individually
        if self.pca_scaling_individual:
            if self.use_pca:
                # scale by singular values to maintain the right variance
                self.scale_factor = 1 / tf.sqrt(
                    tf.cast(pca.singular_values_, dtype=self.dtype_)
                )
            else:
                # scale every feature by its maximum value to avoid numerical issues
                self.scale_factor = 1 / tf.reduce_max(tf.abs(x), axis=0)
        # if no individual scaling is used, scale the whole data set by its maximum value
        else:
            self.scale_factor = 1 / tf.reduce_max(tf.abs(x))

        # apply scaling
        z_pca = z_pca * self.scale_factor

        return z_pca

    def build_pca_decoder(self, z_dec):
        """
        Build a linear decoder which is equivalent to the backprojection of the PCA.

        Parameters
        ----------
        z_dec : tf.Tensor
            Decoded PCA tensor.

        Returns
        -------
        tf.Tensor
            Decoded tensor.
        """
        # apply scaling
        z_dec = z_dec / self.scale_factor
        # pca part
        if self.use_pca:
            x = z_dec @ tf.transpose(self.up)
        # in case no PCA is used, the decoder is just the identity
        else:
            x = z_dec * 1

        return x

    def build_nonlinear_autoencoder(self, z_pca):
        """
        Build a fully connected autoencoder with layers of size layer_sizes.

        Parameters
        ----------
        z_pca : tf.Tensor
            Input to the autoencoder.

        Returns
        -------
        tuple
            Tuple containing encoded and decoded tensors.
        """
        if self.pca_only:
            # Use only PCA and no nonlinear autoencoder
            z = tf.keras.layers.Lambda(lambda x: x * 1.0)(z_pca)
            z_dec = tf.keras.layers.Lambda(lambda x: x * 1.0)(z)
            return z, z_dec

        z = z_pca
        for n_neurons in self.layer_sizes:
            z = tf.keras.layers.Dense(
                n_neurons,
                activation=self.activation,
                activity_regularizer=self.regularizer,
                dtype=self.dtype_,
            )(z)
        z = tf.keras.layers.Dense(
            self.reduced_order, activation="linear", dtype=self.dtype_
        )(z)

        # new decoder
        x_ = z
        for n_neurons in reversed(self.layer_sizes):
            x_ = tf.keras.layers.Dense(
                n_neurons, activation=self.activation, dtype=self.dtype_
            )(x_)
        z_dec = tf.keras.layers.Dense(
            self.pca_order, activation="linear", dtype=self.dtype_
        )(x_)
        return z, z_dec

    @tf.function
    def train_step(self, inputs):
        """
        Perform one training step.

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        dict
            Dictionary containing loss values.
        """
        # perform forward pass, calculate loss and update weights
        rec_loss, dz_loss, dx_loss, reg_loss, loss = self.build_loss(inputs)

        # update loss tracker
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.dz_loss_tracker.update_state(dz_loss)
        self.dx_loss_tracker.update_state(dx_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {
            "loss": self.loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "dz_loss": self.dz_loss_tracker.result(),
            "dx_loss": self.dx_loss_tracker.result(),
            "reg_loss": self.reg_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, inputs):
        """
        Perform one test step.

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        dict
            Dictionary containing loss values.
        """
        rec_loss, dz_loss, dx_loss, reg_loss, loss = self.build_loss(inputs)
        self.loss_tracker.update_state(loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.dz_loss_tracker.update_state(dz_loss)
        self.dx_loss_tracker.update_state(dx_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {
            "loss": loss,
            "rec_loss": rec_loss,
            "dz_loss": dz_loss,
            "dx_loss": dx_loss,
            "reg_loss": reg_loss,
        }

    def calc_latent_time_derivatives(self, x, dx_dt, return_dz_dxr=False):
        """
        Calculate time derivatives of latent variables given the time derivatives of the input variables.

        Parameters
        ----------
        x : array-like
            Input state with shape (n_samples, n_features). Represents the full observed state
            of the system at different time points.

        dx_dt : array-like
            Time derivative of the input state, with shape (n_samples, n_features). Represents
            the rate of change of `x` with respect to time.

        return_dz_dxr : bool, optional (default=False)
            If True, return the Jacobian of the latent variables with respect to the PCA-encoded
            input instead of the latent variables and their derivatives.

        Returns
        -------
        z : np.ndarray
            Latent variables computed from input `x`, with shape (n_samples, latent_dim). Only
            returned if `return_dz_dxr` is False.

        dz_dt : np.ndarray
            Time derivatives of the latent variables, with shape (n_samples, latent_dim). Only
            returned if `return_dz_dxr` is False.

        dz_dxr : tf.Tensor
            Jacobian of the latent variables with respect to the PCA-reduced input. Only returned
            if `return_dz_dxr` is True.
        """

        x = tf.cast(x, self.dtype_)
        dx_dt = tf.expand_dims(tf.cast(dx_dt, dtype=self.dtype_), axis=-1)

        # forward pass of encoder and time derivative of latent variable
        with tf.GradientTape() as t12:
            xr = self.pca_encoder(x)
            t12.watch(xr)
            z = self.nonlinear_encoder(xr)
        dz_dxr = t12.batch_jacobian(z, xr)

        # calculate first time derivative of the latent variable by application of the chain rule
        #   dz_ddt  = dz_dx @ dx_dt
        #           = dz_dxr @ (V^T @ dx_dt)
        try:
            dz_dt = dz_dxr @ tf.expand_dims(self.pca_encoder(dx_dt), axis=-1)
        # for convolutional autoencoder data has another shape and must be vectorized first
        except tf.errors.InvalidArgumentError:
            dz_dxr, dx_dt = self.reshape_conv_data(
                dz_dxr, tf.expand_dims(self.pca_encoder(dx_dt), axis=-1)
            )
            dz_dt = dz_dxr @ dx_dt
        dz_dt = tf.squeeze(dz_dt, axis=2)

        if return_dz_dxr:
            return dz_dxr
        return z.numpy(), dz_dt.numpy()

    def calc_pca_time_derivatives(self, x, dx_dt):
        """
        Calculate time derivatives of PCA variables given the time derivatives of the input variables.

        Parameters
        ----------
        x : array-like
            Full state with shape (n_samples, n_features).
        dx_dt : array-like
            Time derivative of state with shape (n_samples, n_features).

        Returns
        -------
        tuple
            Tuple containing PCA coordinates and their time derivatives.
        """
        x = tf.cast(x, self.dtype_)
        dx_dt = tf.expand_dims(tf.cast(dx_dt, dtype=self.dtype_), axis=-1)

        # forward pass of encoder and time derivative of latent variable
        with tf.GradientTape() as t12:
            z_pca = self.pca_encoder(x)
            t12.watch(z_pca)

        # calculate first time derivative of the latent variable by application of the chain rule
        #   dz_ddt  = dz_dx @ dx_dt
        #           = dz_dxr @ (V^T @ dx_dt)
        dz_pca_dt = tf.expand_dims(self.pca_encoder(dx_dt), axis=-1)
        dz_pca_dt = tf.squeeze(dz_pca_dt, axis=-1)

        return z_pca.numpy(), dz_pca_dt.numpy()

    def calc_physical_time_derivatives(self, z, dz_dt):
        """
        Calculate time derivatives of physical variables given the time derivatives of the latent variables.

        Parameters
        ----------
        z : array-like
            Latent state with shape (n_samples, r).
        dz_dt : array-like
            Time derivative of latent state with shape (n_samples, r).

        Returns
        -------
        tuple
            Tuple containing physical variables and their time derivatives.
        """
        z = tf.cast(z, self.dtype_)
        dz_dt = tf.expand_dims(tf.cast(dz_dt, dtype=self.dtype_), axis=-1)

        # forward pass of encoder and time derivative of latent variable
        with tf.GradientTape() as t12:
            t12.watch(z)
            xr = self.nonlinear_decoder(z)
            x = self.pca_decoder(xr)
        dxr_dz = t12.batch_jacobian(xr, z)

        # calculate first time derivative of the physical variable by application of the chain rule
        #   dx_ddt  = dx_dz @ dz_dt
        #           = V dxr_dz @ dz_dt
        dx_dt = self.pca_decoder(dxr_dz @ dz_dt)

        return x.numpy(), dx_dt.numpy()

    def _get_loss_rec(self, x, dx_dt, u, mu):
        """
        Calculate reconstruction loss of autoencoder.

        Parameters
        ----------
        x : array-like
            Full state with shape (n_samples, n_features).
        dx_dt : array-like
            Time derivative of state with shape (n_samples, n_features).
        u : array-like
            System inputs with shape (n_samples, n_inputs).
        mu : array-like
            System parameters with shape (n_samples, n_params).

        Returns
        -------
        tuple
            Tuple containing individual losses and total loss.
        """
        xr = self.pca_encoder(x)
        z = self.nonlinear_encoder(xr)
        xr_ = self.nonlinear_decoder(z)

        # we only calculate the reconstruction loss for the reconstruction of the pca encoded input
        # because this is the only trainable part and the computational cost is reduced
        rec_loss = self.l_rec * self.compute_loss(None, xr, xr_)

        # for conformity with other get_loss functions return 0 for other losses
        dz_loss = 0
        dx_loss = 0
        # get model losses (e.g. regularization)
        reg_loss = tf.reduce_mean(self.losses) if self.losses else 0
        total_loss = rec_loss + dz_loss + dx_loss + reg_loss

        return rec_loss, dz_loss, dx_loss, reg_loss, total_loss

    def _get_loss(self, x, dx_dt, u, mu):
        """
        Calculate loss.

        Parameters
        ----------
        x : array-like
            Full state with shape (n_samples, n_features).
        dx_dt : array-like
            Time derivative of state with shape (n_samples, n_features).
        u : array-like
            System inputs with shape (n_samples, n_inputs).
        mu : array-like
            System parameters with shape (n_samples, n_params).

        Returns
        -------
        tuple
            Tuple containing individual losses and total loss.
        """
        # time derivative of intermediate latent space
        # dxr_dt = tf.expand_dims(
        #     self.pca_encoder(tf.cast(dx_dt, dtype=self.dtype_)), axis=-1
        # )
        dxr_dt = self.pca_encoder(tf.cast(dx_dt, dtype=self.dtype_))

        # forward pass of encoder and time derivative of latent variable
        with tf.GradientTape() as t12:
            # linear projection of input to intermediate latent space
            xr = self.pca_encoder(x)
            t12.watch(xr)
            # nonlinear mapping of intermediate latent space to latent space
            z = self.nonlinear_encoder(xr)
        dz_dxr = t12.batch_jacobian(z, xr)

        # the second part of the loss calculation is similar for all subclasses and consequently outsourced
        rec_loss, dz_loss, dx_loss = self.get_loss_second_part(
            xr, dz_dxr, dxr_dt, z, u, mu
        )
        # get model losses (e.g. regularization)
        reg_loss = tf.reduce_mean(self.losses) if self.losses else 0
        if self.system_network.losses:
            reg_loss += tf.math.add_n(self.system_network.losses)

        # calculate total loss as weighted sum of individual losses
        total_loss = rec_loss + dz_loss + dx_loss + reg_loss

        return rec_loss, dz_loss, dx_loss, reg_loss, total_loss

    def get_loss_second_part(self, xr, dz_dxr, dxr_dt, z, u, mu):
        """
        Second part of loss calculation (loss calclulation is split into two parts as the second part differs for
        the different autoencoder implementations, while the first part remains the same).

        Parameters
        ----------
        xr : tf.Tensor
            Intermediate latent space tensor.
        dz_dxr : tf.Tensor
            Jacobian of latent variables with respect to intermediate latent space.
        dxr_dt : tf.Tensor
            Time derivative of intermediate latent space.
        z : tf.Tensor
            Latent variables.
        u : tf.Tensor
            System inputs.
        mu : tf.Tensor
            System parameters.

        Returns
        -------
        tuple
            Tuple containing individual losses.
        """

        # calculate first time derivative of the latent variable by application of the chain rule
        #   dz_ddt  = dz_dx @ dx_dt
        #           = dz_dxr @ (V^T @ dx_dt)
        dz_dt = dz_dxr @ tf.expand_dims(dxr_dt, axis=-1)

        # calculate left hand side of ODE system (relevant for descriptor systems)
        # dxr_dt_lhs = self.system_layer.lhs(dxr_dt)
        dxr_dt_lhs = tf.identity(dxr_dt)
        dz_dt_lhs = self.system_layer.lhs(dz_dt[..., 0])

        # system_network approximation of the time derivative of the latent variable
        system_pred = self.system_network([z, u, mu])
        dz_dt_system = tf.expand_dims(system_pred, -1)

        # forward pass of decoder and time derivative of reconstructed variable
        with tf.GradientTape() as t22:
            t22.watch(z)
            xr_ = self.nonlinear_decoder(z)
        # we only calculate this if the loss is not zero to save computational cost
        if self.l_dx > 0.0:
            dxr_dz = t22.batch_jacobian(xr_, z)
            # reshape in case of convolutional autoencoder
            dxr_dz = self.reshape_dxr_dz(dxr_dz)
            # calculate first time derivative of the reconstructed state by application of the chain rule
            dxf_dt = dxr_dz @ dz_dt_system
            dx_loss = self.l_dx * self.compute_loss(None, dxf_dt, dxr_dt_lhs)
        else:
            dx_loss = 0.0
        # calculate losses
        rec_loss = self.l_rec * self.compute_loss(None, xr, xr_)
        dz_loss = (
            self.l_dz
            * self.compute_loss(None, dz_dt_lhs, dz_dt_system)
            / tf.reduce_mean(tf.abs(dz_dt_lhs))
        )
        return rec_loss, dz_loss, dx_loss

    # def compute_loss(self, x, y, y_pred):
    #     """
    #     custom loss function
    #     """
    #     return tf.reduce_mean(self.loss(y, y_pred))

    def reshape_dxr_dz(self, dxr_dz):
        """
        Reshape data for conformity with Convolutional Autoencoder.

        Parameters
        ----------
        dxr_dz : tf.Tensor
            Jacobian of reconstructed state with respect to latent variables.

        Returns
        ----------
        dxr_dz: tf.Tensor
            Same as input
        """
        return dxr_dz

    def vis_modes(self, x, mode_ids=3, latent_ids=None, block=True):
        """
        Visualize the reconstruction of the reduced coefficients of the PCA modes.

        Parameters
        ----------
        x : array-like
            Original dataset.
        mode_ids : int or array-like, optional
            Scalar (plots mode_ids) or array (plots modes with indices from mode_ids).
        latent_ids : int or array-like, optional
            Scalar (plots latent_ids) or array (plots modes with indices from latent_ids).
        block : bool, optional
            Whether to block the display of the plot.

        Returns
        -------
        None
        """
        modes = self.pca_encoder(x)
        if isinstance(
            mode_ids, (list, tuple, np.ndarray)
        ):  # check if n_modes is an array
            n_modes = len(mode_ids)
        else:  # n_modes is scalar: plot modes from 1:n_modes
            n_modes = min(mode_ids, modes.shape[1])
            mode_ids = list(range(n_modes))
        z = self.nonlinear_encoder(modes)
        modes_rec = self.nonlinear_decoder(z)

        # define number of latent coordinate plots (default: plot all latent coordinates)
        if isinstance(
            latent_ids, (list, tuple, np.ndarray)
        ):  # check if n_latent is an array
            n_latent_plots = len(latent_ids)
        elif latent_ids is None:
            # show all latent coordinates
            n_latent_plots = self.reduced_order
            latent_ids = list(range(n_latent_plots))
        else:
            # n_latent is scalar: plot latent coordinates from 1:n_latent
            n_latent_plots = min(latent_ids, self.reduced_order)
            latent_ids = list(range(n_latent_plots))

        # visualize modes in subplots
        n_plots = n_modes + n_latent_plots if self.use_pca else n_latent_plots
        fig, axs = plt.subplots(n_plots, 1, figsize=(10, 10), sharex=True)

        # plot latent variables
        for i in range(n_latent_plots):
            latent_id = latent_ids[i]
            axs[i].plot(z[:, latent_id], color="k")
            axs[i].set_title(f"z_{i}")

        # plot PCA modes (original and reconstructed by latent variables)
        if self.use_pca:
            for i in range(n_modes):
                mode_id = mode_ids[i]
                axs[i + n_latent_plots].plot(modes[:, mode_id], label="Original")
                axs[i + n_latent_plots].plot(
                    modes_rec[:, mode_id], "--", label="Reconstructed"
                )
                axs[i + n_latent_plots].set_title(
                    rf"Mode {mode_id}, $\sigma = {self.singular_values[mode_id]:.4g}$"
                )
                # add legend
                axs[i + n_latent_plots].legend()
        fig.tight_layout()
        plt.show(block=block)

    def encode(self, x):
        """
        Encode full state.

        Parameters
        ----------
        x : array-like
            Full state with shape (n_samples, n_features, n_dof_per_feature).

        Returns
        -------
        z : array-like
            Latent variable with shape (n_samples, reduced_order).
        """
        z = self.encoder(x)
        return z

    def decode(self, z):
        """
        Decode latent variable.

        Parameters
        ----------
        z : array-like
            Latent variable with shape (n_samples, reduced_order).

        Returns
        -------
        x : array-like
            Full state with shape (n_samples, n_features, n_dof_per_feature).
        """
        x_rec = self.decoder(z)
        return x_rec

    def reconstruct(self, x, _=None):
        """
        Reconstruct full state.

        Parameters
        ----------
        x : array-like
            Full state with shape (n_samples, n_features, n_dof_per_feature).

        Returns
        -------
        x_rec : array-like
            Reconstructed full state with shape (n_samples, n_features, n_dof_per_feature).
        """

        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

    # %% time integrator for latent variables (pH structure-preserving integrator)
    def implicit_midpoint(
        self, t0, z0, t_bound, step_size, B=None, u=None, decomp_option=1
    ):
        """
        Calculate time integration of linear ODE through implicit midpoint rule
        ODE system E*dz_dt = A*z + B*u
        Theory:
        We got a pH-system E*Dx = (J-D)*Q*x + B*u
        we define A:=(J-D)*Q and the RHS as f(t,x)
        use the differential slope equation at midpoint
        (x(t+h)-x(t))/h=Dx(t+h/2)=E^-1 * f(t+h/2,x(t+h/2))
        since x(t+h/2) is unknown we use the approximation
        x(t+h/2) = 1/2*(x(t)+x(t+h))
        insert the linear system into the differential equation leads to
        x(t+h) = x(t) + h * E^-1 *(1/2*A*(x(t)+x(t+h))+ B*u(t+h/2))
        reformulate the equation to
        (E-h/2*A)x(t+h) = (E+h/2*A)*x(t) + h*B*u(t+h/2)
        solve the linear equation system, e.g. via LU-decomposition

        Parameters
        ----------
        t0 : float
            Initial time.
        z0 : array-like
            Initial state vector.
        t_bound : float
            End time.
        step_size : float
            Constant step width.
        B : array-like, optional
            Input matrix, default is None (will be set to zero).
        u : callable, optional
            Input function at time midpoints, default is None (will be set to zero).
        decomp_option : int, optional
            Option for decomposition (1-lu_solve), default is 1.

        Returns
        -------
        z : array-like
            Integrated state vector.
        -----------------------------------------------------------------------

        """

        # todo: define self variables that are used

        integrators.implicit_midpoint(
            self.A, self.E, t0, z0, t_bound, step_size, B, u, decomp_option
        )

    def get_projection_properties(self, x=None, x_test=None, file_dir=None):
        """
        Compute and save the projection and Jacobian error.

        Parameters
        ----------
        x : array-like, optional
            Training data.
        x_test : array-like, optional
            Test data.
        file_dir : str, optional
            Directory to save the projection properties.

        Returns
        -------
        tuple
            Tuple containing projection and Jacobian errors for training and test data.
        """

        projection_error, jacobian_error, projection_error_test, jacobian_error_test = [
            None
        ] * 4
        if x is not None:
            projection_error, jacobian_error = self.projection_properties(x)
        if x_test is not None:
            projection_error_test, jacobian_error_test = self.projection_properties(
                x_test
            )
        with open(file_dir, "w") as text_file:
            if x is not None:
                print(f"TRAIN projection error: {projection_error}", file=text_file)
                print(f"TRAIN jacobian error: {jacobian_error}", file=text_file)
            if x_test is not None:
                print(f"TEST projection error: {projection_error_test}", file=text_file)
                print(f"TEST jacobian error: {jacobian_error_test}", file=text_file)

        return (
            projection_error,
            jacobian_error,
            projection_error_test,
            jacobian_error_test,
        )

    def projection_properties(self, x):
        """
        Compute the projection and Jacobian error.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        tuple
            Tuple containing projection error and Jacobian error.
        """
        z = self.encode(x)

        # we only need to evaluate the nonlinear autoencoder since the linear one is a projection per definition

        # forward pass of encoder and time derivative of latent variable
        with tf.GradientTape() as t1:
            t1.watch(z)
            # linear projection of input to intermediate latent space
            v_rec = self.nonlinear_decoder(z)
        # For debug: Patches should be diagonal
        # j = t1.jacobian(v_rec[:10], z[:10])
        # self.plot_as_patches(j)
        jac_z = t1.batch_jacobian(v_rec, z)
        with tf.GradientTape() as t2:
            t2.watch(v_rec)
            # nonlinear mapping of intermediate latent space to latent space
            z_rec = self.nonlinear_encoder(v_rec)
        # For debug: Patches should be diagonal
        # j = t2.jacobian(z_rec[:10], v_rec[:10])
        # self.plot_as_patches(j)
        jac_x = t2.batch_jacobian(z_rec, v_rec)

        # calculate to which extent the autoencoder meets the projection properties
        #   1. enc(dec(z)) == z
        projection_error = tf.reduce_mean(
            tf.square(tf.norm(z - z_rec, axis=1, ord=2))
            / tf.square(tf.norm(z, axis=1, ord=2))
        )
        #   2. jacobian_z(dec(z)) @ jacobian_dec(z)(z) == I
        jacobian_error = tf.reduce_mean(
            tf.square(
                tf.norm(
                    tf.eye(self.reduced_order) - (jac_x @ jac_z), axis=(1, 2), ord=2
                )
            )
        )

        return projection_error, jacobian_error

    @staticmethod
    def plot_as_patches(j):
        """
        Visualize a 4D tensor as a grid of image patches with zero-centered color scale.

        This function rearranges and pads a 4D tensor so that each diagonal of the
        original layout becomes a contiguous patch. It then reshapes the result into a
        single 2D image for visualization.

        This method is taken from the Tensorflow documentation.

        Parameters
        ----------
        j : tf.Tensor
            A 4D tensor with shape (n, m, h, w), where:
                - n is the number of rows of patches,
                - m is the number of columns of patches,
                - h and w are the height and width of each patch.

        Notes
        -----
        - The tensor is transposed so that diagonal elements become contiguous blocks.
        - Padding is applied between patches for visual separation.
        - The final image is plotted using `APHIN.imshow_zero_center`, with a diverging
        colormap centered at zero.
        """
        # Reorder axes so the diagonals will each form a contiguous patch.
        j = tf.transpose(j, [1, 0, 3, 2])
        # Pad in between each patch.
        lim = tf.reduce_max(abs(j))
        j = tf.pad(j, [[0, 0], [1, 1], [0, 0], [1, 1]], constant_values=-lim)
        # Reshape to form a single image.
        s = j.shape
        j = tf.reshape(j, [s[0] * s[1], s[2] * s[3]])
        APHIN.imshow_zero_center(j, extent=[-0.5, s[2] - 0.5, s[0] - 0.5, -0.5])

    def imshow_zero_center(image, **kwargs):
        lim = tf.reduce_max(abs(image))
        plt.imshow(image, vmin=-lim, vmax=lim, cmap="seismic", **kwargs)
        plt.colorbar()
