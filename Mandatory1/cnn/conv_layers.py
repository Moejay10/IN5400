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


    # Retrieve dimensions from input_layers shape (≈1 line)
    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    # Retrieve dimensions from W's shape (≈1 line)
    (num_filters, channels_w, height_w, width_w) = weight.shape

    # Compute the dimensions of the output layer using the formula given in the lecture notes. Hint: use int() to floor. (≈2 lines)
    height_out = int((height_x + 2*pad_size - height_w)/stride) + 1
    width_out =int((width_x + 2*pad_size - width_w)/stride) + 1
    channels_out = num_filters

    # Initialize the output volume Z with zeros. (≈1 line)
    output_layer = np.zeros([batch_size, channels_out, height_out, width_out]) # Should have shape (batch_size, num_filters, height_y, width_y)

    #Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image
    # Create input_layer_pad by padding input_layer
    Input_layer_pad = np.pad(input_layer, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant", constant_values=0)

    for i in range(batch_size):                        # loop over the batch of training examples
        input_layer_pad = Input_layer_pad[i,:,:,:]     # Select ith training example's padded activation
        for h in range(height_out):                           # loop over vertical axis of the output volume
            for w in range(width_out):                       # loop over horizontal axis of the output volume
                for c in range(channels_out):                   # loop over channels (= #filters) of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h*stride
                    vert_end = h*stride + height_w
                    horiz_start = w*stride
                    horiz_end = w*stride + width_w

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    input_slice_prev = input_layer_pad[:, vert_start:vert_end, horiz_start:horiz_end]
                    #print(input_slice_prev.shape)

                    # Element-wise product between a_slice and W. Do not add the bias yet.
                    s = (input_slice_prev * weight[c, :, :, :])
                    # Sum over all entries of the volume s.
                    Z = np.sum(s)
                    # Add bias bias to Z. Get the corresponding bias to the filter so that Z results in a scalar value.
                    Z = Z + bias[c]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    output_layer[i, c, h, w] = Z
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

    # Retrieve dimensions from output_layer_gradient shape
    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    # Retrieve dimensions from input_layer shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    # Retrieve dimensions from weight shape
    num_filters, channels_w, height_w, width_w = weight.shape


    # Initialize gradients of input_layer, weight & bias with the correct shapes
    input_layer_gradient = np.zeros((batch_size, channels_x, height_x, width_x))
    weight_gradient = np.zeros((num_filters, channels_w, height_w, width_w))
    bias_gradient = np.zeros(bias.shape)

    #Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image
    # Create input_layer_pad by padding input_layer
    Input_layer_pad = np.pad(input_layer, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant", constant_values=0)
    Input_layer_gradient_pad = np.pad(input_layer_gradient, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant", constant_values=0)

    for i in range(batch_size):                       # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        input_layer_pad = Input_layer_pad[i]
        input_layer_gradient_pad = Input_layer_gradient_pad[i]

        for h in range(height_y):                  # loop over vertical axis of the output volume
            for w in range(width_y):               # loop over horizontal axis of the output volume
                for c in range(channels_y):        # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h
                    vert_end = vert_start + height_w
                    horiz_start = w
                    horiz_end = horiz_start + width_w

                    # Use the corners to define the slice from a_prev_pad
                    input_layer_slice = input_layer_pad[:, vert_start:vert_end, horiz_start:horiz_end]

                    # Element-wise product between a_slice and W. Do not add the bias yet.
                    s1 = weight[c,:,:,:] * output_layer_gradient[i, c, h, w]
                    s2 = input_layer_slice * output_layer_gradient[i, c, h, w]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    input_layer_gradient_pad[:, vert_start:vert_end, horiz_start:horiz_end] += s1
                    weight_gradient[c,:,:,:] += s2
                    bias_gradient[c] += output_layer_gradient[i, c, h, w]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
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
