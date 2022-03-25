import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """
        For k-nearest neighbors training is just memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y
    
    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test): # for each test sample i
            for j in range(num_train): # applied on all train samples j
                #We compute and store the distance in "dists" variable:
                #We do not need the square **2 as it would be more computationaly expensive
                dists[i,j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j]))) 

        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################

        return dists
    
    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        if num_loops == 0:
          dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
          dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
          dists = self.compute_distances_two_loops(X)
        else:
          raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)
    
    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Hint: Look up the function numpy.argsort.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            #we find the k nearest neighbors (frome the training set) of the ith testing
            #sample and extract the labels of those neighbors: 
            closest_y = np.array(self.y_train[np.argsort(dists[i,:], axis=0)[:k]])
            #Finally, from all the labels extracted from the neighbors, we extract the 
            #most common one by using the argmax() function:
            y_pred[i] = np.bincount(closest_y).argmax()

            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y. You then need to find the #
            # most common label in the list closest_y of labels. Store this label   #
            # in y_pred[i]. Break ties by choosing the smaller                      #
            # label.                                                                #  
            # Hint: You may find these functions useful.                            #
            # numpy.argsort, numpy.argmax, numpy.bincount                           #
            #########################################################################
            pass
            #########################################################################
            #                         END OF YOUR CODE                              #
            #########################################################################

        return y_pred
    
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # We do not need the square **2 as it would be more computationaly expensive
            dists[i,:]=np.sqrt(np.sum(np.square(X[i,:]-self.X_train), axis=1))

        #######################################################################
        # TODO:                                                               #
        # Same as the compute_distances_two_loops function, but this time     #
        # you should only a single loop over the test data.                   #
        #######################################################################
        pass
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #To vectorize: (a-b)**2 = a**2 + b**2 - 2*a*b :
        #axis=1 is the line of the matrix, np.newaxis increase the dimension by one 
        #more dimension, finally .T transposes the matrix
        dists = np.sqrt((X**2).sum(axis=1)[:, np.newaxis] +
                        (self.X_train**2).sum(axis=1) -
                        2*X.dot(self.X_train.T))

        #######################################################################
        # TODO:                                                               #
        # Same as the compute_distances_two_loops function, but this time     #
        # you should NOT use loops.                                           #
        #######################################################################
        pass
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return dists