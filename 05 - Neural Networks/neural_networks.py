import numpy as np

class NeuralNetwork(object):
    def __init__(self, num_layers=2, num_classes=3, hidden_size=10, hidden_activation_fn="relu"):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_activation_fn = hidden_activation_fn
        self.num_classes = num_classes

    def initialize_weights(self, input_dim, std_dev=1e-2):
        """
        Initialize the weights of the model. The weights are initialized
        to small random values and the biases to zero. Weights are stored in 
        the variable dictionary named self.params.
        
        Inputs:
        - input_dim: (int) The dimension D of the input data.
        - std_dev: (float) Controls the standard deviation of the random values.
        """
        
        self.params = {}

        hidden_size = self.hidden_size
        num_classes = self.num_classes
        num_layers = self.num_layers
        #############################################################################
        # TODO: Initialize the weight and bias of every layer. Store the weights W  #
        # of every layer in the dictionary names self.params.                       #
        # For example, Weights and bias of layer 1 will be stored in                #
        # self.params["W1"] and self.params["b1"] respectively.                     #
        #############################################################################
        # bias and weights between input and first hidden layer
        self.params['W'+str(1)] = std_dev * np.random.randn(input_dim, hidden_size)
        self.params['b'+str(1)] = np.zeros(hidden_size)
        # bias and weights between hidden layers
        for i in range(2, num_layers):
            self.params['W'+str(i)] = std_dev * np.random.randn(hidden_size, hidden_size)
            self.params['b'+str(i)] = np.zeros(hidden_size)
        # bias and weights between last hidden layer and the output layer
        self.params['W'+str(num_layers)] = std_dev * np.random.randn(hidden_size, num_classes)
        self.params['b'+str(num_layers)] = np.zeros(num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        


    def fully_connected_forward(self, X, W, b):
        """
        Computes the forward pass of a fully connected layer.
        
        A fully connected / affine / linear / dense layer applies a linear transformation
        of the incoming data: Wx + b.

        Inputs:
        - X: A numpy array of shape (N, D)
        - W: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output of shape (N, M)
        - cache: (X, W, b)
        """
        
        #############################################################################
        # TODO: Implement the forward pass of a fully connected layer and store     #
        # the variables needed for the backward pass (gradient computation)         #
        # as a tuple inside cache.                                                  #
        #############################################################################
        out = X.dot(W)+b
        cache = (X, W, b)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return out, cache

    def fully_connected_backward(self, dUpper, cache):
        """
        Computes the backward pass for a fully connected layer layer.

        Inputs:
        - dUpper: Gradient of shape (N, M), coming from the upper layer.
        - cache: Tuple of:
            - X: A numpy array of shape (N, D)
            - W: A numpy array of weights, of shape (D, M)
            - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - dX: Gradient with respect to X, of shape (N, D)
        - dW: Gradient with respect to W, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        X, W, b = cache
        dX, dW, db = None, None, None
        #############################################################################
        # TODO: Implement the affine backward pass.                                 #
        #############################################################################
        dW = X.T @ dUpper
        dX = dUpper @ W.T
        db = np.sum(dUpper, axis=0)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dX, dW, db

    def sigmoid_forward(self, x):
        """
        Computes the forward pass for sigmoid activation function.

        Input:
        - x: Inputs, a numpy array of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        
        #############################################################################
        # TODO: Implement the Sigmoid forward pass.                                 #
        #############################################################################
        # Sigmoid function is defined as : sig(x) = 1/( 1+exp(-x) )
        out = 1/(1+np.exp(-x))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = out
        return out, cache


    def sigmoid_backward(self, dUpper, cache):
        """
        Computes the backward pass for a sigmoid activation function.

        Input:
        - dUpper: Upstream derivatives coming from the upper layers.
        - cache: Input x, of same shape as dUpper.

        Returns:
        - dsigmoid: Gradient with respect to x
        """
        out = cache
        #############################################################################
        # TODO: Implement the backward pass for the sigmoid function.               #
        #############################################################################
        # sig'(x) = sig(x) * (1 - sig(x))
        # if Y is sig(x), to find sig' depending on Y, we obtain sig'(Y) = Y * (1-Y)
        dsigmoid = out * (1 - out) * dUpper
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dsigmoid
    
    def softmax(self,x):
        
        """
        Compute the softmax function for each row of the input x.

        Inputs:
        - x: A numpy array of shape (N, C) containing scores for each class; there are N
          examples each of dimension C.

        Returns:
        probs: A numpy array of shape (N, C) containing probabilities for each class.
        """

        #############################################################################
        # TODO: Implement the softmax function.                                     #
        #############################################################################
        probs = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            # Scale x[i] to lower values, thanks to the mathematical property.
            x[i] -= np.max(x[i])
            # Softmax(x)[i] = exp(x[i])/sum(exp(x[i]))
            probs[i] = np.exp(x[i])/np.sum(np.exp(x[i]))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return probs

    def softmax_cross_entropy_loss(self, scores, labels):
        """
        Jointly computes the softmax and cross entropy loss. This function should return
        the loss and its gradient with respect to the scores.

        Inputs:
        - scores: A numpy array of shape (N, C) containing scores for each class; there are N
          examples each of dimension C.
        - labels: A numpy array of shape (N,) containing the indices of the correct class for
          each example.

        Returns:
        loss: A scalar value corresponding to the softmax cross entropy loss
        dloss: A numpy array of shape (N, C) containing the gradients of the loss with respect
            to the scores.
        """

        #############################################################################
        # TODO: Compute for the softmax cross entropy loss                          #
        #############################################################################
        N, C = scores.shape
        probs = self.softmax(scores)
        # loss = -sum(log(probability of X[i] belonging to labels[i])
        loss = 0
        for i in range(N):
            loss -= np.log(probs[i,labels[i]])
        loss = loss/N
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        #############################################################################
        # TODO: Compute for the gradients of the loss with respect to the scores    #
        #############################################################################
        # Derivatives(loss)[i] = probs[i] - labels[i]
        dloss = np.zeros((N, C))
        for i in range(N):
            y = np.zeros(C)
            y[labels[i]] = 1
            dloss[i] = probs[i] - y
        dloss /= N
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return loss, dloss

    def network_forward(self, X):
        """
        This functions performs the forward pass which computes for the class scores given
        the input.

        Inputs:
        - X: A numpy array of shape (N, D) containing the data; there are N
          samples each of dimension D.
        
        Returns:
        scores: A numpy array of shape (N, C) containing class scores.
        cache_list: A list containing the cached values to be used on the backward pass.
        """
        
        #############################################################################
        # TODO: Perform a forward pass on the network and store the caches of       #
        # each layer inside the cache_list                                          #
        #############################################################################
        num_layers = self.num_layers
        cache_list = []
        # cache_list[0] stores the cache from fully_connected_forward:
        cache_list.append([])
        # cache_list[1] stores the cache from sigmoid_forward:
        cache_list.append([])
        # Propa through the hidden layers:
        for i in range(1, num_layers):
            out, cache = self.fully_connected_forward(X, self.params['W'+str(i)], self.params['b'+str(i)])
            cache_list[0].append(cache)
            X, cache = self.sigmoid_forward(out)
            cache_list[1].append(cache)
        # No sigmoid activation function through the output layer
        scores, cache = self.fully_connected_forward(X, self.params['W'+str(num_layers)], self.params['b'+str(num_layers)])
        cache_list[0].append(cache)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores, cache_list

    def network_backward(self, dloss, cache_list):
        """
        This functions performs the backward pass which computes for the gradients of the
        loss with respect to every parameter.

        Inputs:
        - dloss: A numpy array of shape (N, C) corresponding to the gradient of the 
            loss with respect to the scores outputted during the forward pass.
        - cache_list: A list of the cached values during the forward pass.
        
        Returns:
        grads: A dictionary containing the gradients of every parameter. For example, the gradients
            of the weights and bias of the first layer is stored in grads["W1"] and grads["b1"]
            respectively.
        """
        
        #############################################################################
        # TODO: Implement the backward pass.                                        #
        #############################################################################
        grads = {}
        num_layers = self.num_layers
        # Propa through the output layer
        dX, grads['W'+str(num_layers)], grads['b'+str(num_layers)] = self.fully_connected_backward(dloss, cache_list[0][num_layers-1])
        # Propa through the hidden layers
        for i in range(num_layers-1, 0, -1):
            dsigmoid = self.sigmoid_backward(dX, cache_list[1][i-1])
            dX, grads['W'+str(i)], grads['b'+str(i)] = self.fully_connected_backward(dsigmoid, cache_list[0][i-1])
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return grads

    def loss(self, X, y=None, lambda_reg=0.0):
        """
        Compute the loss and gradients for an iteration.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the ground truth value for X[i].
        - lambda_reg: Regularization strength.

        Returns:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        N, D = X.shape
        # Compute the forward pass
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        scores, cache_list = self.network_forward(X)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        
        #############################################################################
        # TODO: Compute for the loss. This should include L2 regularization for     #
        # the weights of each layer.                                                #
        #############################################################################
        
        loss, dloss = self.softmax_cross_entropy_loss(scores, y)
        
        # Ridge Regression is defined as: 1/2 * lambda * Frobenius_norm(W)
        W = self.params['W'+str(1)]
        for i in range(2,self.num_layers+1):
            W = np.append(W, self.params['W'+str(i)])
            
        R = 0.5 * lambda_reg * np.linalg.norm(W)
        
        loss += R
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        
        #############################################################################
        # TODO: Compute the derivatives of the weights and biases. Store the        #
        # results in the grads dictionary. For example, grads['W1'] should store    #
        # the gradient on the weights W of the first layer, and be a matrix of      #
        # same size.                                                                #
        #############################################################################
        grads = self.network_backward(dloss, cache_list)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def tanh_forward(self, x):
        """
        Computes the forward pass for the tanh activation function.

        Input:
        - x: Inputs, a numpy array of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        
        #############################################################################
        # TODO: Implement the tanh forward pass.                                    #
        #############################################################################
        # tanh(x) = (1-exp(2x)) / (1+exp(2x))
        exp = np.exp(-2*x)
        out = (1-exp) / (1+exp)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = out
        return out, cache


    def tanh_backward(self, dUpper, cache):
        """
        Computes the backward pass for tanh activation function.

        Input:
        - dUpper: Upstream derivatives coming from the upper layers.
        - cache: Input x, of same shape as dUpper.

        Returns:
        - dtanh: Gradient with respect to x
        """
        out = cache
        #############################################################################
        # TODO: Implement the tanh backward pass.                                   #
        #############################################################################
        # if Y is tanh(x)=2*sig(2x), we want to find the derivative of tanh depending on Y
        # with tanh'(x) = 4*sig(2x)*sig'(2x) = (2* (2*sig(2x)-1) +2) * (1- (2*sig(2x)-1) /2)
        dtanh = (2 * out + 2) * (1-((out+1)/2)) * dUpper
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dtanh
    
    def relu_forward(self, x):
        """
        Computes the forward pass of a rectified linear unit (ReLU).

        Input:
        - x: A numpy array / matrix of any shape

        Returns a tuple of:
        - out: A numpy array / matrix of the same shape as x
        - cache: x
        """
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        #############################################################################
        # ReLU(x) = max(0,x)
        out = np.maximum(0, x)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = x
        return out, cache


    def relu_backward(self, dUpper, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dUpper: Upstream derivatives coming from the upper layers.
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        x = cache
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        #############################################################################
        # If x <= 0, then drelu(x) = 0
        # Else, drelu(x) = 1
        drelu = (x>0) * dUpper
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return drelu
    

    def train_step(self, X, y, learning_rate=1e-3, lambda_reg=1e-5, batch_size=200):

        num_train, dim = X.shape

        indices = np.random.choice(num_train,batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]

        loss, grads = self.loss(X_batch, y=y_batch, lambda_reg=lambda_reg)

        for i in range(self.num_layers):
            self.params["W"+str(i+1)] += - learning_rate * grads["W"+str(i+1)]
            self.params["b"+str(i+1)] += - learning_rate * grads["b"+str(i+1)]

        return loss, grads


    def train(self, X, y, learning_rate=1e-3, lambda_reg=0.0, num_iters=100, std_dev=1e-2,
            batch_size=200, verbose=False, one_step=False):
        """
        Train Linear Regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N, 1) containing the ground truth values.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        self.initialize_weights(dim, std_dev)

        loss_history = []
        for it in range(num_iters):

            loss, grads = self.train_step(X, y, learning_rate, lambda_reg, batch_size)

            if it % 100 == 0:
                loss_history.append(np.squeeze(loss))

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X, return_scores=False):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - return_scores: A flag that decides whether to return the scores or not.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        scores, cache_list = self.network_forward(X)
        probs = self.softmax(scores)
        prediction = np.argmax(probs, axis=1)


        if return_scores:
            return prediction, scores
        else:
            return prediction



