import numpy as np


def distance(X,Y):

  """

  distance (X,Y) returns a matrix
  that contains the Euclidean Distance
  between vectors in X and vectors
  in Y. Requires that vectors are same dimension.

  X: nxd matrix containing n vectors as rows, each of dimension d.
  Y: mxd matrix containing m vectors as rows, each of dimension d.

  Returns a matrix nxm for which (i,j) is the distance
  between vector x_i in X and vector y_i in Y.

  """

  n,dx = X.shape
  m,dy = Y.shape

  assert(dx == dy), "Dimensions are not the same"

  # here in order to be quick, we use define G = X.T @ X
  # and let our distance matrix be diag(G) + diag(G)^T - 2G
  # https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf

  G = X.T @ X

  return np.diag(G) + np.diag(G).T -2*G