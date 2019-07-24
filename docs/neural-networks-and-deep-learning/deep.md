## Implement

```python
def L_layer_model(
        X, Y, layers_dims,
        learning_rate = 0.0075, num_iterations = 3000, print_cost=False
    ):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat),
         of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size,
                   of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model.
                  They can then be used to predict.
    """

    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)


    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    return parameters
```

```python
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions
    of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1",
                  "b1", ..., "WL", "bL", L = len(layer_dims) - 1

    Examples:
    layer_dims = [5,4,3] gives W1, b1, W2, b2
    """
```

```python
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    computation
    call 'linear_activation_forward'

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
              every cache of linear_activation_forward() (there are L-1 of
              them, indexed from 0 to L-1)
    """
```

```python
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    call 'linear_forward'
    called by 'L_model_forward'

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of
              previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer,
         size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
         activation -- the activation to be used in this layer, stored
         as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the
         post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
```

```python
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    called by 'linear_activation_forward'

    Arguments:
    A -- activations from previous layer (or input data): (size of previous
         layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size
         of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation
         parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for
             computing the backward pass efficiently
    """
```

```python
def compute_cost(AL, Y):
    """
    Implement the cost function defined.

    Arguments:
    AL -- probability vector corresponding to your label predictions,
          shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat),
         shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
```

```python
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the
    [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    call 'linear_activation_backward'

    Arguments:
    AL -- probability vector, output of the forward propagation
          (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu"
                (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid"
                (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
```

```python
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    call 'linear_backward'
    called by 'L_model_backward'

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for
             computing backward propagation efficiently
    activation -- the activation to be used in this layer,
                  stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the
               previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
          same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
          same shape as b
    """
```

```python
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation
    for a single layer (layer l)
    called by 'linear_activation_backward'

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output
          (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation
             in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the
               previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
          same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
          same shape as b
    """
```

```python
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of
             L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
```
