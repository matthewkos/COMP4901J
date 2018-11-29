from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma : scale for batch_norm shifting
    - beta: bias for batch_norm shifting
    - bn_param
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass

    """
    a, fc_cache = affine_forward(x, w, b)
    bn, batchnorm_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, batchnorm_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, batchnorm_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn, batchnorm_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_relu_dropout_forward(x, w, b, dropout_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - dropout_param:  mode, p, seed
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass

    """
    a, fc_cache = affine_forward(x, w, b)
    activated, relu_cache = relu_forward(a)
    out, dropout_cache = dropout_forward(activated, dropout_param)
    cache = (fc_cache, relu_cache, dropout_cache)
    return out, cache


def affine_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache, dropout_cache = cache
    dactivated = dropout_backward(dout, dropout_cache)
    da = relu_backward(dactivated, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_batchnorm_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma : scale for batch_norm shifting
    - beta: bias for batch_norm shifting
    - bn_param
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features
    - dropout_param: mode, p, seed
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass

    """
    a, fc_cache = affine_forward(x, w, b)
    bn, batchnorm_cache = batchnorm_forward(a, gamma, beta, bn_param)
    activated, relu_cache = relu_forward(bn)
    out, dropout_cache = dropout_forward(activated, dropout_param)
    cache = (fc_cache, batchnorm_cache, relu_cache, dropout_cache)
    return out, cache


def affine_batchnorm_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, batchnorm_cache, relu_cache, dropout_cache = cache
    dactivated = dropout_backward(dout, dropout_cache)
    dbn = relu_backward(dactivated, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn, batchnorm_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # W1: (input_dim, hidden_dim)   b1: (hidden_dim,)
        # W2: (hidden_dim, num_classes) b2: (num_classes,)
        W1 = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, hidden_dim))
        b1 = np.zeros((hidden_dim,))
        W2 = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes))
        b2 = np.zeros((num_classes,))
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        layer1, l1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        layer2, l2_cache = affine_forward(layer1, self.params['W2'], self.params['b2'])
        scores = layer2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1'])
                                  + np.sum(self.params['W2'] * self.params['W2']))
        dlayer1, dw2, db2 = affine_backward(dout, l2_cache)
        grads['W2'] = dw2 + self.reg * l2_cache[1]
        grads['b2'] = db2
        dx, dw1, db1 = affine_relu_backward(dlayer1, l1_cache)
        grads['W1'] = dw1 + self.reg * l1_cache[0][1]
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        dims = [input_dim] + hidden_dims + [num_classes]
        # W1 = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, hidden_dims[0]))
        # b1 = np.zeros((hidden_dims[0],))
        # self.params={'W1': W1, 'b1': b1}
        assert self.num_layers == len(dims) - 1
        if self.use_batchnorm:
            for l in range(len(dims) - 1):
                self.params['W{}'.format(l + 1)] = np.random.normal(loc=0, scale=weight_scale,
                                                                    size=(dims[l], dims[l + 1]))
                self.params['b{}'.format(l + 1)] = np.zeros((dims[l + 1],))
                if l < len(dims) - 2:
                    self.params['gamma{}'.format(l + 1)] = np.ones((dims[l + 1],))
                    self.params['beta{}'.format(l + 1)] = np.zeros((dims[l + 1],))
        else:
            for l in range(len(dims) - 1):
                self.params['W{}'.format(l + 1)] = np.random.normal(loc=0, scale=weight_scale,
                                                                    size=(dims[l], dims[l + 1]))
                self.params['b{}'.format(l + 1)] = np.zeros((dims[l + 1],))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        layers = {}
        caches = {}
        out = X
        if not self.use_batchnorm and not self.use_dropout:
            for i in range(self.num_layers - 1):
                layers[i + 1], caches[i + 1] = affine_relu_forward(out, self.params['W{}'.format(i + 1)],
                                                                   self.params['b{}'.format(i + 1)])
                out = layers[i + 1]
            layers[self.num_layers], caches[self.num_layers] = affine_forward(out, self.params[
                'W{}'.format(self.num_layers)], self.params['b{}'.format(self.num_layers)])
        elif self.use_batchnorm and not self.use_dropout:
            for i in range(self.num_layers - 1):
                layers[i + 1], caches[i + 1] = affine_batchnorm_relu_forward(
                    out, self.params['W{}'.format(i + 1)],
                    self.params['b{}'.format(i + 1)],
                    self.params['gamma{}'.format(i + 1)],
                    self.params['beta{}'.format(i + 1)],
                    self.bn_params[i])
                out = layers[i + 1]
            layers[self.num_layers], caches[self.num_layers] = affine_forward(out, self.params[
                'W{}'.format(self.num_layers)], self.params['b{}'.format(self.num_layers)])
        elif not self.use_batchnorm and self.use_dropout:
            for i in range(self.num_layers - 1):
                layers[i + 1], caches[i + 1] = affine_relu_dropout_forward(out, self.params['W{}'.format(i + 1)],
                                                                           self.params['b{}'.format(i + 1)],
                                                                           self.dropout_param)
                out = layers[i + 1]
            layers[self.num_layers], caches[self.num_layers] = affine_forward(out, self.params[
                'W{}'.format(self.num_layers)], self.params['b{}'.format(self.num_layers)])
        elif self.use_batchnorm and self.use_dropout:
            for i in range(self.num_layers - 1):
                layers[i + 1], caches[i + 1] = affine_batchnorm_relu_dropout_forward(
                    out, self.params['W{}'.format(i + 1)],
                    self.params['b{}'.format(i + 1)],
                    self.params['gamma{}'.format(i + 1)],
                    self.params['beta{}'.format(i + 1)],
                    self.bn_params[i],
                    self.dropout_param)
                out = layers[i + 1]
            layers[self.num_layers], caches[self.num_layers] = affine_forward(out, self.params[
                'W{}'.format(self.num_layers)], self.params['b{}'.format(self.num_layers)])

        scores = layers[self.num_layers]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        L2_reg = 0
        # dlayer = dout
        if not self.use_batchnorm and not self.use_dropout:
            L2_reg += np.sum(self.params['W{}'.format(self.num_layers)] ** 2)
            dlayer, dw, db = affine_backward(dout, caches[self.num_layers])
            grads['W{}'.format(self.num_layers)] = dw + self.reg * caches[self.num_layers][1]
            grads['b{}'.format(self.num_layers)] = db
            for i in range(self.num_layers - 1, 0, -1):
                L2_reg += np.sum(self.params['W{}'.format(i)] ** 2)
                dlayer, dw, db = affine_relu_backward(dlayer, caches[i])
                grads['W{}'.format(i)] = dw + self.reg * caches[i][0][1]
                grads['b{}'.format(i)] = db
            loss += 0.5 * self.reg * L2_reg
        elif self.use_batchnorm and not self.use_dropout:
            L2_reg += np.sum(self.params['W{}'.format(self.num_layers)] ** 2)
            dlayer, dw, db = affine_backward(dout, caches[self.num_layers])
            grads['W{}'.format(self.num_layers)] = dw + self.reg * caches[self.num_layers][1]
            grads['b{}'.format(self.num_layers)] = db
            for i in range(self.num_layers - 1, 0, -1):
                L2_reg += np.sum(self.params['W{}'.format(i)] ** 2)
                dlayer, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dlayer, caches[i])
                grads['W{}'.format(i)] = dw + self.reg * caches[i][0][1]
                grads['b{}'.format(i)] = db
                grads['gamma{}'.format(i)] = dgamma
                grads['beta{}'.format(i)] = dbeta
            loss += 0.5 * self.reg * L2_reg
        elif not self.use_batchnorm and self.use_dropout:
            L2_reg += np.sum(self.params['W{}'.format(self.num_layers)] ** 2)
            dlayer, dw, db = affine_backward(dout, caches[self.num_layers])
            grads['W{}'.format(self.num_layers)] = dw + self.reg * caches[self.num_layers][1]
            grads['b{}'.format(self.num_layers)] = db
            for i in range(self.num_layers - 1, 0, -1):
                L2_reg += np.sum(self.params['W{}'.format(i)] ** 2)
                dlayer, dw, db = affine_relu_dropout_backward(dlayer, caches[i])
                grads['W{}'.format(i)] = dw + self.reg * caches[i][0][1]
                grads['b{}'.format(i)] = db
            loss += 0.5 * self.reg * L2_reg
        elif self.use_batchnorm and self.use_dropout:
            L2_reg += np.sum(self.params['W{}'.format(self.num_layers)] ** 2)
            dlayer, dw, db = affine_backward(dout, caches[self.num_layers])
            grads['W{}'.format(self.num_layers)] = dw + self.reg * caches[self.num_layers][1]
            grads['b{}'.format(self.num_layers)] = db
            for i in range(self.num_layers - 1, 0, -1):
                L2_reg += np.sum(self.params['W{}'.format(i)] ** 2)
                dlayer, dw, db, dgamma, dbeta = affine_batchnorm_relu_dropout_backward(dlayer, caches[i])
                grads['W{}'.format(i)] = dw + self.reg * caches[i][0][1]
                grads['b{}'.format(i)] = db
                grads['gamma{}'.format(i)] = dgamma
                grads['beta{}'.format(i)] = dbeta
            loss += 0.5 * self.reg * L2_reg
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
