import numpy as np
import preprocess as pp
import naive_bayes
import logisticregression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf

model = KNeighborsClassifier(n_neighbors=40)
model2 = GaussianNB()
model3 = MultinomialNB()
model4 = OneVsRestClassifier(LogisticRegression(max_iter=100))

print("done0")
X, Y, labels = pp.vectorize("data/arcDatasetSmall2", False)

print("done1")

train_data, train_labels, test_data, test_labels = pp.data_split(X, Y, 0.8)

print("done2")

#model2.fit(train_data, train_labels)

#model.fit(train_data, train_labels)
#model4.fit(train_data, train_labels)
#print(model.score(test_data, test_labels))
#print(model2.score(test_data, test_labels))
#print(model4.score(test_data, test_labels))
#model3.fit(train_data, train_labels)
#print(model3.score(test_data, test_labels))


# print(naive_bayes_c.nbanalysis_mle(
# train_data, train_labels, test_data, test_labels))
# print(naive_bayes_c.nbanalysis_map(
# train_data, train_labels, test_data, test_labels))

logisticregression.predict_LR(
    logisticregression.adagrad, X, Y, test_data, 1.0, 100)

# print(logisticregression.LR_analysis(logisticregression.adagrad, train_data,
#      train_labels, test_data, test_labels, 1.0, 100))
# print(logisticregression.LR_analysis(logisticregression.adagrad, train_data,
#      train_labels, test_data, test_labels, 1.0, 100, mat=True))
# print(logisticregression.LR_analysis(logisticregression.logistic_regression, train_data,
#                                    train_labels, test_data, test_labels, 1.0, 100))
# print(logisticregression.LR_analysis(logisticregression.logistic_regression, train_data,
#                                  train_labels, test_data, test_labels, 1.0, 100, mat=True))
