import datetime
import os

import tensorflow as tf


def callbacks(
    weight_path,
    monitor="loss",
    tensorboard=True,
    log_dir=None,
    earlystopping=False,
    patience=100,
    save_many_weights=False,
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

    save_many_weights : bool, optional
        If True, adds a custom callback (`WeightBeforeEpochCallback`) that saves model weights at regular
        intervals (e.g., every 5 epochs). These are saved separately from the best model checkpoint.
        Default is False.

    Returns:
    --------
    list of tf.keras.callbacks.Callback
        A list of TensorFlow Keras callbacks configured according to the provided parameters.
    """
    if ".weights.h5" not in weight_path:
        weight_path = os.path.join(weight_path, ".weights.h5")
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            weight_path,
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

    if save_many_weights:
        callback_list.append(
            WeightBeforeEpochCallback(
                os.path.join(weight_dir, "before_ep{epoch}.weights.h5"),
                monitor="val_loss",
                save_best_only=False,
                save_weights_only=True,
            )
        )

    return callback_list


class WeightBeforeEpochCallback(tf.keras.callbacks.ModelCheckpoint):
    """
    A custom TensorFlow Keras callback for saving model weights at the beginning of specific epochs.

    This callback extends the `ModelCheckpoint` class and overwrites the `on_epoch_begin`
    method to allow saving the model at the start of every 5th epoch (or at a frequency specified by
    `save_freq`). The callback saves the model weights before each epoch begins, ensuring that
    intermediate model states can be captured during training.

    The `on_epoch_end` method updates the current epoch but does not perform additional actions by default.

    Notes
    -----
    - The model is saved when `epoch % 5 == 0` by default, but this can be adjusted if needed.
    - This callback overrides `on_epoch_begin` and `on_epoch_end` methods from `ModelCheckpoint`.

    Parameters
    ----------
    save_freq : str or int, optional
        Frequency to save the model. Default is "epoch", meaning the model will be saved every epoch.
        A custom frequency (e.g., every 5th epoch) can be specified by modifying `on_epoch_begin`.
    """

    # overwrite epoch_on_begin
    def on_epoch_begin(self, epoch, logs=None):
        if self.save_freq == "epoch":
            if epoch % 5 == 0:
                self._save_model(epoch=epoch, batch=None, logs=logs)
                print(f"I'm saving the model for epoch {epoch}.")

    def on_epoch_end(self, epoch, logs=None):
        self._current_epoch = epoch
        # print("I'm at the end of an epoch.")
