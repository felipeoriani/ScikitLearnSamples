# A simple XOR Classifier using Multi-Layer Perceptron Neural Network

import sklearn.neural_network as nn
import matplotlib.pyplot as plt

# define the input set and expected output
inputSet = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputSet = [0, 1, 1, 0]

# create the classifier based on Multi-Layer perceptron with the following arguments
model = nn.MLPClassifier(activation='tanh', max_iter=200, solver='lbfgs')

# train the classifier
model = model.fit(inputSet, outputSet)

# print some results
print('score: ', model.score(inputSet, outputSet))
print('predictions: ', model.predict(inputSet)) 
print('expected: ', outputSet)

# some asserts for the trained model
assert model.predict([[0, 0]]) == [0]
assert model.predict([[0, 1]]) == [1]
assert model.predict([[1, 0]]) == [1]
assert model.predict([[1, 1]]) == [0]

# plot a error chat
# plt.plot(model.loss_curve_, label='loss')
# plt.show()

# Reference
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html#