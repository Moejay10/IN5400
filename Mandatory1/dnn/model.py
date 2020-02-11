#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image Analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1
    params = {}

    n_layers = len(conf['layer_dimensions'])

    mu = 0


    # Weights and Biases
    for i in range(0, n_layers-1):
        sigma2 = float(2/conf['layer_dimensions'][i])
        n_neurons_prev = conf['layer_dimensions'][i]
        n_neurons_next = conf['layer_dimensions'][i+1]
        params["W_"+ str(i+1)] = np.sqrt(sigma2)*np.random.randn(n_neurons_prev, n_neurons_next) + mu
        params["b_"+ str(i+1)] = np.zeros((n_neurons_prev, 1))




    return params


def RelU(Z):
    return (Z>0)*Z


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        return RelU(Z)
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 b)

    Z_copy = Z.copy()

    # Normalization trick to avoid numerical instability,
    # Trick 1
    Z_copy -= np.max(Z_copy, axis=0, keepdims=True) # max of every sample

    # Compute loss (and add to it, divided later).
    # Denominator of softmax function.
    sum_Z_copy = np.sum(np.exp(Z_copy), axis=0, keepdims=True)

    # Computing log(softnmax)
    # Numerator - denominator of sotmax function
    # Trick 2.
    log_p = Z_copy - np.log(sum_Z_copy)

    # The softmax for all classes C
    loss = np.exp(log_p)

    # Sum up crossentropy loss of the sample i
    # to become N samples minibatch crossentropy loss
    # for each class C
    #loss = np.sum(p, axis=1, keepdims=True)
    #loss = np.exp(f)/sum_f


    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c)

    n_layers = len(conf['layer_dimensions'])

    features = {}

    features['Z_' + str(1)] = np.dot(params['W_' + str(1)].T, X_batch) + params['b_' + str(1)]

    for l in range(1, n_layers-1):
        features['A_' + str(l)] = activation(features['Z_' + str(l)], 'relu')
        features['Z_' + str(l+1)] = np.dot(params['W_' + str(l+1)].T, features['A_' + str(l)]) + params['b_' + str(l+1)]
        print(np.dot(params['W_' + str(l+1)].T, features['A_' + str(l)]) + params['b_' + str(l+1)])

    Y_proposed = softmax(features['Z_' + str(n_layers-1)])
    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3
    cost = None
    num_correct = None

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    if activation_function == 'relu':
        return None
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)
    grad_params = None
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    updated_params = None
    return updated_params
