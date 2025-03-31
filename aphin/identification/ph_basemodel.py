from abc import ABC
import os
import logging
import inspect
import datetime
import pickle
import copy
import time
import tensorflow as tf

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class PHBasemodel(tf.keras.Model, ABC):
    """
    Base model for port-Hamiltonian identification networks.
    """

    def _init_to_config(self, init_locals):
        """
        Save the parameters with which the model was initialized, except for the data itself.

        Parameters
        ----------
        init_locals : dict
            Local variables from the __init__ function.
        """

        sig = inspect.signature(self.__init__)
        orig_keys = [param.name for param in sig.parameters.values()]

        # we don't want to save the data itself, so we remove x, u and mu
        banned_keys = ["x", "mu", "u", "kwargs"]

        keys = [name for name in orig_keys if name not in banned_keys]
        values = [init_locals[name] for name in keys]

        for i, key in enumerate(["x", "u", "mu"]):
            if key not in orig_keys:
                keys.insert(i, key)
                values.insert(i, None)

        # handle kwargs
        if "kwargs" in orig_keys:
            kwarg_keys = [
                key for key in init_locals["kwargs"].keys() if key not in banned_keys
            ]
            kwarg_values = [init_locals["kwargs"][key] for key in kwarg_keys]
            # add Nones for the data
            for i, key in enumerate(["x", "u", "mu"]):
                if key not in init_locals["kwargs"].keys():
                    kwarg_keys.insert(i, key)
                    kwarg_values.insert(i, None)
            kwarg_dict = dict(zip(kwarg_keys, kwarg_values))
            values.append(kwarg_dict)
            keys.append("kwargs")

        init_dict = dict(zip(keys, values))
        # deep copy kwargs so we can manipulate them without changing the original
        if "kwargs" in init_dict.keys():
            init_dict["kwargs"] = copy.deepcopy(init_dict["kwargs"])
        # we don't want to save the data itself
        init_dict["x"] = None
        init_dict["mu"] = None
        if "x" in init_dict["kwargs"].keys():
            init_dict["kwargs"]["x"] = None
        if "mu" in init_dict["kwargs"].keys():
            init_dict["kwargs"]["mu"] = None
        self.config = init_dict

    def save(self, path: str = None):
        """
        Save the model weights and configuration to a given path.

        Parameters
        ----------
        path : str, optional
            Path to the folder where the model should be saved, by default None.

        Returns
        -------
        None
        """
        if path is None:
            path = (
                f"results/saved_models/{self.__class__.__name__}/"
                f'{datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")}/'
            )
        weights_path = os.path.join(path, "weights/")
        model_path = os.path.join(path, f"model_config.pkl")
        self.save_weights(weights_path)
        # self.config['class_name'] = self.__class__.__name__
        with open(model_path, "wb") as outp:  # Overwrites any existing file.
            pickle.dump(self.config, outp)

    def compute_loss(self, x, y, y_pred):
        return tf.reduce_mean(self.loss(y, y_pred))

    @staticmethod
    def load(
        ph_network,
        x=None,
        u=None,
        mu=None,
        path: str = None,
        kwargs_overwrite: dict = None,
    ):
        """
        Load the model from the given path.

        Parameters
        ----------
        ph_network : callable
            The port-Hamiltonian network to be loaded.
        x : array-like, optional
            Data needed to initialize the model, by default None.
        u : array-like, optional
            Control inputs, by default None.
        mu : array-like, optional
            Parameters used to create the model the first time, by default None.
        path : str, optional
            Path to the model, by default None.
        kwargs_overwrite : dict, optional
            Additional kwargs to overwrite the config, by default None.

        Returns
        -------
        PHBasemodel
            Loaded model.
        """

        weights_path = os.path.join(path, "weights/")
        model_path = os.path.join(path, f"model_config.pkl")
        with open(model_path, "rb") as file:  # Overwrites any existing file.
            init_dict = pickle.load(file)
        init_dict["x"] = x
        init_dict["u"] = u
        init_dict["mu"] = mu
        kwargs = init_dict.pop("kwargs")
        # overwrite the kwargs with values from kwargs_overwrite
        if kwargs_overwrite is None:
            kwargs_overwrite = {}
        kwargs.update(kwargs_overwrite)
        if "x" in kwargs:
            kwargs.pop("x")
        if "u" in kwargs:
            kwargs.pop("u")
        if "mu" in kwargs:
            kwargs.pop("mu")
        loaded_model = ph_network(**init_dict, **kwargs)
        loaded_model.load_weights(weights_path)

        return loaded_model

    def fit(self, x, y=None, validation_data=None, **kwargs):
        """
        Wrapper for the fit function of the Keras model to flatten the data if necessary.

        Parameters
        ----------
        x : array-like
            Training data.
        y : array-like, optional
            Target data, by default None.
        validation_data : tuple or array-like, optional
            Data for validation, by default None.
        **kwargs : dict
            Additional arguments for the fit function.

        Returns
        -------
        History
            A `History` object. Its `History.history` attribute is a record of training loss values and metrics values
             at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        if validation_data is not None:
            if isinstance(validation_data, tuple):
                validation_x, validation_y = validation_data
            else:
                validation_x = validation_data
                validation_y = None
            for i, x_ in enumerate(validation_x):
                try:
                    validation_x[i] = x_
                except tf.errors.InvalidArgumentError:
                    validation_x[i] = x_
            validation_data = (validation_x, validation_y)

        import time

        start = time.time()
        history = super(PHBasemodel, self).fit(
            x, y, validation_data=validation_data, **kwargs
        )
        end = time.time()
        time = end - start
        time_per_epoch = time / len(history.history["loss"])
        logging.info(
            f"Training took {time} s with an average of {time_per_epoch} s per epoch."
        )
        # save the time per epoch in the history
        history.history["time_per_epoch"] = time_per_epoch
        history.history["time"] = time

        return history

    def get_trainable_weights(self):
        """
        Get the trainable weights of the model.

        Returns
        -------
        list
            List of trainable weights.
        """
        return self.system_network.trainable_weights

    def build_loss(self, inputs):
        """
        Split input into state, its derivative, and the parameters, perform the forward pass, calculate the loss,
        and update the weights.

        Parameters
        ----------
        inputs : list of array-like
            Input data.

        Returns
        -------
        list
            List of loss values.
        """

        # split inputs into state, its derivative and the parameters if available:
        # split inputs into state, its derivative and the parameters if available
        x, dx_dt, u, mu = self.split_inputs(inputs)

        # forward pass
        with tf.GradientTape() as tape:
            # only perform reconstruction if no identification loss is used
            losses = self.get_loss(x, dx_dt, u, mu)

            # split trainable variables for autoencoder and dynamics so that you can use seperate optimizers
            trainable_weights = self.get_trainable_weights()
            grads = tape.gradient(losses[-1], trainable_weights)
            self.optimizer.apply_gradients(zip(grads, trainable_weights))

        return losses

    def split_inputs(self, inputs):
        """
        Split inputs into state, its derivative, and the parameters.

        Parameters
        ----------
        inputs : list of array-like
            Input data.

        Returns
        -------
        tuple
            Tuple containing state, its derivative, control inputs, and parameters.
        """
        # split inputs into state, its derivative and the parameters if available:
        # initialize variables as empty tensors
        x, dx_dt, u, mu = [None] * 4

        # first order system with parameter / inputs
        if len(inputs[0]) == 4:
            [x, dx_dt, u, mu] = inputs[0]
        # first order system without parameter / inputs
        elif len(inputs[0]) == 3:
            [x, dx_dt, u] = inputs[0]
            mu = tf.zeros((0, 0))
        # first order system without parameter / inputs
        elif len(inputs[0]) == 2:
            [x, dx_dt] = inputs[0]
            u = tf.zeros((0, 0))
            mu = tf.zeros((0, 0))

        # ensure that the data is of type float32
        x = tf.cast(x, self.dtype)
        dx_dt = tf.cast(dx_dt, self.dtype)
        if u is not None:
            u = tf.cast(u, self.dtype)
        if mu is not None:
            mu = tf.cast(mu, self.dtype)

        return x, dx_dt, u, mu

    def get_system_weights(self):
        """
        Get the weights of the system identification part of the model.

        Returns
        -------
        list
            List of system weights.
        """
        return self.system_network.trainable_weights
