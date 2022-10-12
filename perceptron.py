import numpy as np
import time
import preprocess as pp
# Here X: nxd is our data, Y: 1xn are our labels, and k is the upper
# bound on iterations

# returns weight vector w: 1xd and bias b to use as linear classifier

# https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975


def perceptron(X, Y, k):
    n, d = X.shape
    # have b as our last entry
    w = np.zeros(d+1)
    count = 0
    lst = np.arange(n)  # we must randomize our data each cycle

    X = np.hstack((X, np.ones((n, 1))))

    while count < k:
        m = 0
        np.random.shuffle(lst)
        for i in lst:
            x = X[i]
            if Y[i] * (w @ x.T) <= 0:  # improper classification
                w = w + (Y[i] * x.flatten())
                m = m + 1
        if m == 0:
            break

        count = count + 1

    b = w[-1]
    w = w[:-1]

    return w, b


def perceptron_classify(train_data, train_labels, test_data, test_labels, k):

    start_time = time.time()
    w, b = perceptron(train_data, train_labels, k)
    total_time = time.time() - start_time

    # puts number of misclassifications over number of tests
    predictions = np.sign((w @ test_data.T) + b).flatten()
    print(predictions.shape)

    out = (test_labels.size - np.count_nonzero(test_labels -
           predictions)) / test_labels.size

    return out, total_time


X, Y, labels = pp.vectorize("data/arcDatasetBin")

train_data, train_labels, test_data, test_labels = pp.data_split(X, Y, 0.8)

print(test_data.shape)

# print(perceptron_classify(np.array([[1, 1], [2, 1], [3, 5], [4, 5], [6, 5],
# [-1, -1], [-2, -1], [-3, -5], [-4, -5], [-6, -5]]), np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]), np.array([[800, 900], [1, 1], [-800, -900], [-1, -1]]), np.array([1, 1, -1, -1]), 1000000))
print(perceptron_classify(train_data, train_labels, test_data, test_labels, 10000))
