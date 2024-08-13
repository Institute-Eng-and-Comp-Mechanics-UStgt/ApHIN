import datetime
import os

import tensorflow as tf


def callbacks(
    weight_dir,
    monitor="loss",
    tensorboard=True,
    log_dir=None,
    earlystopping=False,
    patience=100,
):
    """
    Create a list of TensorFlow Keras callbacks for model training.

    This function generates a set of callbacks to be used during model training in TensorFlow Keras.
    It includes mandatory callback for saving the best model weights and optional callbacks for
    TensorBoard logging, early stopping, and learning rate scheduling.

    Parameters:
    -----------
    weight_dir : str
        Path to the directory where the best model weights will be saved.

    monitor : str, optional
        The metric to monitor for saving the best model and early stopping. Default is "loss".

    tensorboard : bool, optional
        If `True`, enables TensorBoard logging. Default is `True`.

    log_dir : str, optional
        Path to the directory where TensorBoard logs will be saved. If `None`, a default log directory
        with a timestamp is created. Default is `None`.

    earlystopping : bool, optional
        If `True`, enables early stopping based on the monitored metric. Default is `False`.

    patience : int, optional
        Number of epochs with no improvement after which training will be stopped if early stopping is enabled.
        Default is `100`.


    Returns:
    --------
    list of tf.keras.callbacks.Callback
        A list of TensorFlow Keras callbacks configured according to the provided parameters.
    """
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(weight_dir, ".weights.h5"),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    if tensorboard:
        if log_dir is None:
            datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = "logs/fit/" + datetime_str
        callback_list.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        )

    if earlystopping:
        callback_list.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=patience, restore_best_weights=True
            )
        )

    return callback_list
