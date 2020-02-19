#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Implementation of convolution forward and backward pass"""

import numpy as np



def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # TODO: Task 2.1


    # Initialize dimensions from input_layers shape
    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    # Initialize dimensions from weigths shape
    (num_filters, channels_w, height_w, width_w) = weight.shape

    # Compute the dimensions of the output layer using the formula given in the notebook
    height_out = int((height_x + 2*pad_size - height_w)/stride) + 1
    width_out =int((width_x + 2*pad_size - width_w)/stride) + 1
    channels_out = num_filters

    # Initialize the output with zeros
    output_layer = np.zeros([batch_size, channels_out, height_out, width_out]) # Should have shape (batch_size, num_filters, height_y, width_y)

    #Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image
    # Create input_layer_pad by padding input_layer
    Input_layer_pad = np.pad(input_layer, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant", constant_values=0)

    for i in range(batch_size):
        # Select ith batch from Input_layer_pad
        input_layer_pad = Input_layer_pad[i,:,:,:]

        for h in range(height_out):
            for w in range(width_out):
                for c in range(channels_out):

                    # Find the start and end
                    y_start = h*stride
                    y_end = h*stride + height_w
                    x_start = w*stride
                    x_end = w*stride + width_w

                    # Use the corners to define the input
                    input = input_layer_pad[:, y_start:y_end, x_start:x_end]

                    # Element-wise product between input and weights
                    z = (input * weight[c, :, :, :])
                    output = np.sum(z)
                    # Add bias bias to output
                    output = output + bias[c]
                    # Compute output_layer
                    output_layer[i, c, h, w] = output
    ### END CODE HERE ###

    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return output_layer


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # TODO: Task 2.2

    # Initialize dimensions from output_layer_gradient shape
    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    # Initialize dimensions from input_layer shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    # Initialize dimensions from weight shape
    num_filters, channels_w, height_w, width_w = weight.shape


    # Initialize gradients of input_layer, weight & bias with the correct shapes
    input_layer_gradient = np.zeros((batch_size, channels_x, height_x, width_x))
    weight_gradient = np.zeros((num_filters, channels_w, height_w, width_w))
    bias_gradient = np.zeros(bias.shape)

    #Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image
    # Create input_layer_pad by padding input_layer
    Input_layer_pad = np.pad(input_layer, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant", constant_values=0)
    Input_layer_gradient_pad = np.pad(input_layer_gradient, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant", constant_values=0)

    for i in range(batch_size):
        # select ith batch from Input_layer_padded and Input_layer_gradient_pad
        input_layer_pad = Input_layer_pad[i,:,:,:]
        input_layer_gradient_pad = Input_layer_gradient_pad[i,:,:,:]

        for h in range(height_y):
            for w in range(width_y):
                for c in range(channels_y):

                    # Find the start and end
                    y_start = h
                    y_end = y_start + height_w
                    x_start = w
                    x_end = x_start + width_w

                    # Use the corners to define the input_layer_
                    input_layer_ = input_layer_pad[:, y_start:y_end, x_start:x_end]

                    # Element-wise product between input_, weights and output
                    s1 = weight[c,:,:,:] * output_layer_gradient[i, c, h, w]
                    s2 = input_layer_ * output_layer_gradient[i, c, h, w]

                    # Update gradients
                    input_layer_gradient_pad[:, y_start:y_end, x_start:x_end] += s1
                    weight_gradient[c,:,:,:] += s2
                    bias_gradient[c] += output_layer_gradient[i, c, h, w]

        # Compute input_layer_gradient
        input_layer_gradient[i, :, :, :] = input_layer_gradient_pad[:, pad_size:-pad_size, pad_size:-pad_size]

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return input_layer_gradient, weight_gradient, bias_gradient


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
