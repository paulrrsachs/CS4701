import numpy as np
import time
from numba import jit


def distance(X, Y):
    # distance (X,Y) returns a matrix
    # that contains the Euclidean Distance
    # between vectors in X and vectors
    # in Y. Requires that vectors are same dimension.

    # X: nxd matrix containing n vectors as rows, each of dimension d.
    # Y: mxd matrix containing m vectors as rows, each of dimension d.

    # Returns a matrix nxm for which (i,j) is the distance
    # between vector x_i in X and vector y_i in Y.

    # This works for both large test vector sets and for a single
    # vector, and is useful for K-NN

    n, dx = X.shape
    m, dy = Y.shape

    #assert (dx == dy), "Dimensions do not matched"

    # For our distance matrix D[i][j]^2 = (x_i - y_j)(x_i-y_j)^T
    # as it is the distance between vector x_i in X and y_j in Y squared
    # here in order to be quick, we use the following expression to find D

    # D^2 = S + R -2 * X @ Y.T

    # Where S[i][j] = x_i.T @ x_i for all j and R[i][j] = y_j.T @ y_j for all j
    # Here both S and R are nxm

    # https://nenadmarkus.com/p/all-pairs-euclidean/

    # Here we take X @ X.T, and multiply it element wise by the identity matrix
    # so that we just have the diagonal x_i.T @ x_i

    # We then matrix multiply by a row of ones so that  we get
    # a row of values x_i.T @ x_i. We transpose this so it is a column
    # and then tile it to get our matrix S
    S = np.tile((np.ones((1, n)) @ ((X @ X.T)*np.identity(n))).T, (1, m))

    # This case is similar to the previous, but we keep y_i.T @ y_i as
    # a row and tile it down.

    R = np.tile((np.ones((1, m)) @ ((Y @ Y.T)*np.identity(m))), (n, 1))
    D = np.sqrt(S + R - 2 * X @ Y.T)

    return D

# data: nxd matrix of our training data

# labels: 1xn matrix containing the labels for each vector in the data

# test: mxd matrix containing the vectors we wish to classify,
# is 1xd if it is a single vector.

# k: number of neighbors we are checking, k < n

# returns 1xm matrix containg the labels we predict for our data


def knn_classify(data, labels, test, k):

    # D[i][j] is the distance between data vector x_i and the test vector y_j

    n, d1 = data.shape
    m, d2 = test.shape
    assert k < n

    D = distance(data, test)

    # Now we sort along axis 0 (rows), that way D[:][j] is  sorted column
    # of the closest data vectors to our test vector y_j

    dists = np.sort(D, 0)

    # we also get the indices of these data vectors so that we can
    # tell which data vectors are closest to y_j

    indicies = np.argsort(D, 0)

    # Now we resize the matrices so each column is of length k, only recording
    # the k nearest neighbors

    dists = np.resize(dists, (k, m))
    indicies = np.resize(indicies, (k, m))
    # print(indicies)
    # print(labels)

    # kxm matrix where labels[i][j] is the label of the ith nearest neighbor
    # of test vector j
    labels = labels[indicies]

    # Now we classify each vector by finding the mode
    # of values along each column

    def classify(arr):
        uniq, counts = np.unique(arr, return_counts=True)
        return uniq[np.argmax(counts)]

    X = np.apply_along_axis(classify, 0, labels)

    return X


# data: nxd matrix of our training data

# data_labels: 1xn matrix containing the labels for each vector in the data

# test: mxd matrix containing the vectors we wish to classify,
# is 1xd if it is a single vector.

# test_labels: 1xn matrix containing the labels for each vector in the test

# k: number of neighbors we are checking, k < n

# returns accuracy of model and the time it takes to predict classifications


# @jit(nopython=True)
def knn_analysis(data, data_labels, test, test_labels, k):

    start_time = time.time()
    predictions = knn_classify(data, data_labels, test, k)
    total_time = time.time() - start_time

    # puts number of misclassifications over number of tests

    out = (test_labels.size - np.count_nonzero(test_labels -
           predictions)) / test_labels.size

    return out, total_time

# Same as knn_analysis but for a single vector, returns the prediction
# rather than the accuracy


def knn_single(data, data_labels, test, test_label, k):
    start_time = time.time()
    prediction = knn_classify(data, data_labels, test, k)
    total_time = time.time() - start_time

    return prediction.flatten(), total_time
