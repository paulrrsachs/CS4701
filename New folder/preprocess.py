from asyncore import file_dispatcher
from cProfile import label
from genericpath import isdir
from re import I
import numpy as np
import scipy.misc as sc
import os
from PIL import Image, ImageOps


# preprocess image collection at path, grayscaling and converting
# images to size=(x,y)

def preprocess(size, path):
    for sub_dir in os.listdir(path):
        for file in os.listdir(path + "/" + sub_dir):
            im = Image.open(path + "/" + sub_dir + "/" + file)
            imResize = im.resize(size)
            final = ImageOps.grayscale(imResize)
            final.save(path + "/" + sub_dir + "/" + file)


# takes in path to directory containing data, must contain a test.jpg
# in order set the dimensions of the data

# returns a  matrix X nxd where n is our number of images, d is the dimension
# also returns Y 1xn being the integer label for each image
# Also returns int_to_labels mapping labels of images to strings
def vectorize(path, is_binary):

    # get dimension of a single image
    d = np.asarray(Image.open(
        path + "/test.jpg")).flatten().size

    # n is number of data vectors
    n = 0

    for root_dir, cur_dir, files in os.walk(path):
        n = len(files) + n

    labels_to_int = {}
    int_to_labels = {}

    # X is our data and Y is our labels
    X = np.zeros((n, d))
    Y = np.zeros(n)
    i = 0
    if is_binary:
        i = -1
    j = 0
    for sub_dir in os.listdir(path):
        labels_to_int[sub_dir] = i
        int_to_labels[i] = sub_dir
        # for correct subdirectories
        if sub_dir != "test.jpg":
            for file in os.listdir(path + "/" + sub_dir):
                im = np.asarray(Image.open(
                    path + "/" + sub_dir + "/" + file)).flatten() * (1.0 / 255)
                # label image as correct class
                Y[j] = i
                X[j] = im
                j = j+1

            # change class label after iterating through directory
            if i == -1:
                i = 1
            else:
                i = i+1
    return X, Y, int_to_labels


# processes a single image into a vector

# path: path to image
# size: size of image after conversion (width, heught)

# outputs a numpy vector

def proc_single(path, size):

    im = Image.open(path)
    imResize = im.resize(size)
    final = ImageOps.grayscale(imResize)
    return np.vstack([np.asarray(final).flatten()*(1.0 / 255), np.zeros(size[0] * size[1])])


# data is nxd and labels is 1xn and k is in [0,1] and is the percent
# of the data used for training

# returns the training data, the corresponding training labels
# the test data and the corresponding test labels
def data_split(data, labels, k):
    n, d = data.shape

    # append labels to each corresponding data vector

    #X = np.append(data, labels.T, axis=1)
    indicies = np.random.permutation(n)
    data = data[indicies]
    labels = labels[indicies]

    split = int(np.ceil(k * n))

    # takes first k*10% of data in permuation, must have at least 10 elements
    # in each class to work

    train_data = data[:split][:]
    test_data = data[split:][:]

    train_labels = labels[:split]
    test_labels = labels[split:]

    #rain_data = train[:][:-1]
    #train_labels = train[:][-1].flatten()

    #test_data = test[:][:-1]
    #test_labels = test[:][-1].flatten()

    return train_data, train_labels, test_data, test_labels


# preprocess((96, 96), "data/arcDatasetProc2")  # DO NOT RUN MORE THAN ONCE
#X, Y, labels = vectorize("data/arcDatasetProc")

# print(X.shape)
# print(Y.shape)
# print(labels)
