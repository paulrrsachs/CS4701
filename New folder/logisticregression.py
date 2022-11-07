from xmlrpc.client import boolean
import numpy as np
import time
import pickle

# w: 1xd weight vector
# x: 1xd vector we are lebeling
# y: scalar representing label

# implements the sigmoid function


def sigmoid(w, x, y):
    return 1.0 / (1+np.exp(-y*(w.T @ x)))

# w: 1xd weight vector
# X: nxd matrix of data
# neg_labes: 1xn vector representing labels, negated

# implements the sigmoid function


def sigmoid_mat(w, X, neg_labels):
    return 1.0 / (1+np.exp(neg_labels*(X @ w.T)))
# w: 1xd weight vector
# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label

# returns a gradient with respect to w


def gradient(w, X, Y):
    n, d = X.shape
    grad = np.zeros(d)

    for i in range(n):
        grad = grad + (1 - sigmoid(w, X[i], Y[i])) * \
            sigmoid(w, X[i], Y[i]) * (-Y[i] * X[i])
    return grad

# w: 1xd weight vector
# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing labels, negated

# returns a gradient with respect to w


def gradient_mat(w, X, neg_labels):
    grad = (1 - sigmoid_mat(w, X, neg_labels)) @ (X.T * neg_labels).T
    return grad

# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label
# alpha: learning rate
# limit: upper bound on number of iterations
# delta: how small we want the gradient norm to be
# mat: whether or not mat version should be used instad of loops

# implements plain linear regrssion


def logistic_regression(X, Y, alpha, limit, delta=1e-03, mat=False):
    n, d = X.shape
    # adds 1 to each vector so we can learn our intercept
    X = np.hstack((X, np.ones((n, 1))))
    w = np.zeros(d+1)
    iter = 0
    neg_labels = -Y

    converged = False
    g = np.zeros(d+1)

    while not converged and iter < limit:
        if mat:
            g = gradient_mat(w, X, neg_labels)
        else:
            g = gradient(w, X, Y)

        w = w - alpha * g
        iter = iter + 1

        if np.linalg.norm(g) < delta:
            converged = True

    return w


# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label
# alpha: learning rate
# epsilon: causes learning rate to decrease
# limit: number of iterations
# delta: how small we want the gradient norm to be
# mat: Should matrix version of gradient be used

# implements adagrad to find a weight vector w


def adagrad(X, Y, alpha, limit, epsilon=0.1, delta=1e-03, mat=False):
    n, d = X.shape
    # adds 1 to each vector so we can learn our intercept
    X = np.hstack((X, np.ones((n, 1))))
    w = np.zeros(d+1)
    z = np.zeros(d+1)
    iter = 0
    neg_labels = -Y

    converged = False
    ep = epsilon * np.ones(d+1)

    while not converged and iter < limit:
        if mat:
            g = gradient_mat(w, X, neg_labels)
        else:
            g = gradient(w, X, Y)

        z = z + np.square(g)
        w = w - alpha * np.divide(g, np.sqrt(z+ep))
        iter = iter + 1

        if np.linalg.norm(g) < delta:
            converged = True

    np.save("models/LB.npy", w)

    return w

# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label
# alpha: learning rate
# epsilon: causes learning rate to decrease

# tests: mxd matrix of m test vectors

# func: method of optimization
# limit: upper bound on number of iterations
# mat: should mat version be used for sigmoid and gradient
# load: can we load classifiers that were already trained

# classify each test vector using one vs all


def predict_LR(func, X, Y, tests, alpha, limit, mat=False, load=False):

    m, d = tests.shape
    n, d1 = X.shape

    tests = np.hstack((tests, np.ones((m, 1))))

    classes = np.unique(Y)

    classifiers = dict()

    # find one-vs all classifiers for each class
    # if we have a pre-trained classifer, load
    if load:
        classifiers = np.load("models/LR.npy", allow_pickle=True)
        # with open('models/LR.pickle', 'rb') as handle:
        #classifiers = pickle.Unpickler(handle).load()

    else:
        for i in classes:
            indicies = np.nonzero(Y - i)
            # set all non class i vectors to be -1
            Y_i = np.ones(n)
            Y_i[indicies] = -1
            classifiers[i] = func(X, Y_i, alpha, limit, mat=mat)
        # save classifer

        np.save("models/LR.npy", classifiers)
        # with open('models/LR.pickle', 'wb') as handle:
        # pickle.dump(classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    preds = np.zeros(m)

    for x in range(m):
        probs = dict()
        for i in classes:
            if load:
                probs[i] = classifiers.item().get(i).T @ tests[x]
            else:
                probs[i] = classifiers[i].T @ tests[x]
            preds[x] = max(probs, key=probs.get)

    return preds

# X: nxd matrix of data vectors
# Y: 1xn vector of scalar representing label
# alpha: learning rate
# epsilon: causes learning rate to decrease
# func: method of optimization
# limit: upper bound on number of iterations
# mat: bool, should we use matrix version

# tests: mxd matrix of m test vectors

# classify each test vector using one vs all


def LR_analysis(func, data, data_labels, test, test_labels, alpha, limit, mat=False):

    start_time = time.time()
    predictions = predict_LR(func,
                             data, data_labels, test, alpha, limit, mat=mat)
    total_time = time.time() - start_time

    # puts number of misclassifications over number of tests

    out = (test_labels.size - np.count_nonzero(test_labels -
                                               predictions)) / test_labels.size

    return out, total_time

# predict single image
