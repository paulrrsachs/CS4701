from xmlrpc.client import boolean
import numpy as np
import time

# w: 1xd weight vector
# x: 1xd vector we are lebeling
# y: scalar representing label

# implements the sigmoid function


def sigmoid(w, x, y):
    return 1.0 / (1+np.exp(-y*(w.T @ x)))
# w: 1xd weight vector
# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label

# returns a gradient with respect to w


def gradient(w, X, Y):
    n, d = X.shape
    grad = np.zeros(d)

    for i in range(n):
        grad = grad + (1 - sigmoid(w, X[i], Y[i])) * (-Y[i] * X[i])
    return grad

# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label
# alpha: learning rate

# implements plain linear regrssion


def logistic_regression(X, Y, alpha, limit):
    n, d = X.shape
    # adds 1 to each vector so we can learn our intercept
    X = np.hstack((X, np.ones((n, 1))))
    w = np.zeros(d+1)
    iter = 0

    converged = False

    while not converged and iter < limit:
        g = gradient(w, X, Y)

        w_n = w - alpha * g

        if np.allclose(w, w_n[0]):
            converged = True
        w = w_n
        iter = iter + 1
    return w


# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label
# alpha: learning rate
# epsilon: causes learning rate to decrease
# limit: number of iterations

# implements adagrad to find a weight vector w


def adagrad(X, Y, alpha, limit, epsilon=0.1):
    n, d = X.shape
    # adds 1 to each vector so we can learn our intercept
    X = np.hstack((X, np.ones((n, 1))))
    w = np.zeros(d+1)
    z = np.zeros(d+1)
    iter = 0

    converged = False
    ep = np.full((1, d+1), epsilon)

    while not converged and iter < limit:
        g = gradient(w, X, Y)

        z = z + np.square(g)

        w_n = w - alpha * np.divide(g, np.sqrt(z+ep))

        if np.allclose(w, w_n[0]):
            converged = True
        w = w_n[0]
        iter = iter + 1

    return w

# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label
# alpha: learning rate
# epsilon: causes learning rate to decrease

# tests: mxd matrix of m test vectors

# func: method of optimization
# limit: upper bound on number of iterations

# classify each test vector using one vs all


def predict_LR(func, X, Y, tests, alpha, limit):

    m, d = tests.shape
    n, d1 = X.shape

    tests = np.hstack((tests, np.ones((m, 1))))

    classes = np.unique(Y)
    print(classes)

    classifiers = dict()

    # find one-vs all classifiers for each class

    for i in classes:
        indicies = np.nonzero(Y - i)
        # set all non class i vectors to be -1
        Y_i = np.ones(n)
        Y_i[indicies] = -1
        classifiers[i] = func(X, Y_i, alpha, limit)

    preds = np.zeros(m)

    for x in range(m):
        probs = dict()
        for i in classes:
            probs[i] = classifiers[i].T @ tests[x]
        preds[x] = max(probs, key=probs.get)

    return preds

# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label
# alpha: learning rate
# epsilon: causes learning rate to decrease
# func: method of optimization
# limit: upper bound on number of iterations

# tests: mxd matrix of m test vectors

# classify each test vector using one vs all


def LR_analysis(func, data, data_labels, test, test_labels, alpha, limit):

    start_time = time.time()
    predictions = predict_LR(func,
                             data, data_labels, test, alpha, limit)
    total_time = time.time() - start_time

    # puts number of misclassifications over number of tests

    out = (test_labels.size - np.count_nonzero(test_labels -
                                               predictions)) / test_labels.size

    return out, total_time
