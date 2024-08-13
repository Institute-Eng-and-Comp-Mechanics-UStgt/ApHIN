import tensorflow as tf


def _pad_last2d(x, paddings, mode="CONSTANT"):
    """
    Pad the last two dimensions of a tensor of batched matrices.

    This function applies `tf.pad` to the last two axes/dimensions of a
    tensor of batched matrices.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor to be padded. It should be a tensor containing batched matrices.
    paddings : array-like
        A 2D array specifying the number of values to pad for each dimension. Should have the shape [2, 2],
        where paddings[i] = [pad_before, pad_after] specifies how much padding to add before and after the ith dimension.
    mode : str, optional
        The padding mode to use. Default is "CONSTANT". Other modes include "REFLECT", "SYMMETRIC", etc.
        See `tf.pad` documentation for more details on padding modes.

    Returns
    -------
    tf.Tensor
        A tensor with the same rank as the input tensor `x`, with the last two dimensions padded according
        to the specified `paddings`.

    Notes
    -----
    The function constructs the padding tensor to apply `tf.pad` only to the last two dimensions of the input tensor.
    Leading dimensions are padded with zeros to preserve the shape of the input tensor.

    See Also
    --------
    tf.pad : TensorFlow function for padding tensors.
    """

    leading_paddings = tf.zeros([tf.rank(x) - len(paddings), 2], dtype=tf.int32)
    new_paddings = tf.concat([leading_paddings, paddings], axis=0)
    return tf.pad(x, new_paddings, mode)


def _transpose_last2d(x):
    """
    Transpose the last two dimensions of a tensor of batched matrices.

    This function applies `tf.transpose` to the last two axes/dimensions of a
    tensor of batched matrices.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor to be transposed. It should be a tensor containing batched matrices.

    Returns
    -------
    tf.Tensor
        A tensor with the same rank as the input tensor `x`, with the last two dimensions transposed.
    """

    trailing_axes = [-1, -2]
    leading = tf.range(tf.rank(x) - len(trailing_axes))
    trailing = trailing_axes + tf.rank(x)
    new_order = tf.concat([leading, trailing], axis=0)
    return tf.transpose(x, perm=new_order)
