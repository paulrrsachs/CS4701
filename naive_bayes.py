from cmath import pi
from statistics import variance
import numpy as np
import time

# https://www.cs.toronto.edu/~urtasun/courses/CSC411_Fall16/09_naive_bayes.pdf

# X: nxd matrix of n data vectors
# Y 1xn vector of n classifications
# inputss: mxd matrix of m test vectors


def nbpredict_mle(X, Y, tests):
    classes = np.unique(Y)
    m, d1 = tests.shape
    n, d = X.shape
    # find a mean and variance for each classification and dimension
    means = dict()
    variance = dict()

    # for each class, find mean of each dimension
    for i in classes:
        class_means = dict()
        class_variance = dict()
        # get indicies for vectors which are classified this way
        temp = Y - i
        # zeroes are the correct classification
        indices = np.where(temp == 0)[0]
        X_i = X[indices]
        # get mean of each dim

        for j in range(0, d):
            # find variance for each dimension
            class_means[j] = np.mean(X_i[:, j])
            class_variance[j] = np.var(X_i[:, j])
        means[i] = class_means
        variance[i] = class_variance

    preds = np.zeros(m)

    for x in range(0, m):
        probs = dict()
        for i in classes:
            sum = 0
            for j in range(0, d):
                sum = sum - 0.5 * (
                    np.log(2*pi) - np.log(variance[i][j]) -
                    (((X[x, j] - means[i][j])**2) / variance[i][j]))

            probs[i] = sum
        preds[x] = max(probs, key=probs.get)

    return preds


def nbpredict_map(X, Y, tests):
    classes = np.unique(Y)
    m, d1 = tests.shape
    n, d = X.shape
    # find a mean and variance for each classification and dimension
    means = dict()
    variance = dict()
    priors = dict()

    for i in classes:
        priors[i] = 0
        class_means = dict()
        class_variance = dict()
        # get indicies for vectors which are classified this way
        temp = Y - i
        # zeroes are the correct classification
        indices = np.where(temp == 0)[0]
        X_i = X[indices]
        # get mean of each

        for j in range(0, d):
            # find variance for each dimension
            class_means[j] = np.mean(X_i[:, j])
            class_variance[j] = np.var(X_i[:, j])
        means[i] = class_means
        variance[i] = class_variance

    # set priors

    for i in range(n):
        priors[Y[i]] = priors[Y[i]] + 1

    for i in classes:
        priors[i] = priors[i] * 1.0 / n
    preds = np.zeros(m)

    for x in range(0, m):
        probs = dict()
        for i in classes:
            sum = 0
            for j in range(0, d):
                sum = sum - 0.5 * (
                    np.log(2*pi) - np.log(variance[i][j]) -
                    (((X[x, j] - means[i][j])**2) / variance[i][j]))

            probs[i] = sum + priors[i]
        preds[x] = max(probs, key=probs.get)

    return preds


def nbanalysis_mle(data, data_labels, test, test_labels):

    start_time = time.time()
    predictions = nbpredict_mle(data, data_labels, test)
    total_time = time.time() - start_time

    # puts number of misclassifications over number of tests

    out = (test_labels.size - np.count_nonzero(test_labels -
                                               predictions)) / test_labels.size

    return out, total_time


def nbanalysis_map(data, data_labels, test, test_labels):

    start_time = time.time()
    predictions = nbpredict_map(data, data_labels, test)
    total_time = time.time() - start_time

    # puts number of misclassifications over number of tests

    out = (test_labels.size - np.count_nonzero(test_labels -
                                               predictions)) / test_labels.size

    return out, total_time
