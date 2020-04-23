# A simple XOR Classifier using Multi-Layer Perceptron Neural Network

import sklearn.neural_network as nn
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

# define the input set and expected output
inputSet = [[0, 0], [1, 0], [0, 1], [1, 1]]
outputSet = [0, 1, 1, 0]

# create the classifier based on Multi-Layer perceptron with the following arguments
model = nn.MLPClassifier(activation='tanh', max_iter=2000, hidden_layer_sizes=(5,))

# train the classifier
model = model.fit(inputSet, outputSet)

# results
score = model.score(inputSet, outputSet)
pred = model.predict(inputSet)
mse = metrics.mean_squared_error(outputSet, pred)
cm = metrics.confusion_matrix(outputSet, pred, labels=[0, 1])
cms = str(np.matrix(cm))

# print some metrics

print('score:', score)
print('predictions: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*pred))
print('expected:', outputSet)
print('mean squared error: {:.4f}'.format(mse))
print('confusion matrix:')
print(cms)

# some asserts for the trained model
assert model.predict([[0, 0]]) == [0]
assert model.predict([[0, 1]]) == [1]
assert model.predict([[1, 0]]) == [1]
assert model.predict([[1, 1]]) == [0]
assert mse == 0

# plot a error chat (not available on MLPClassifier.solver='lbfgs')
if hasattr(model, 'loss_curve_'):
    color = '#1a7eb8'
    font = {'family': 'serif', 'color': color, 'weight': 'normal', 'size': 16}
    plt.title('Loss Curve - MSE: {:.4f}'.format(mse), font)
    plt.ylabel('error', font)
    plt.xlabel('max_iter', font)
    plt.plot(model.loss_curve_, label = 'loss', color = color)
    plt.text(0.5, 0.1, 'by Felipe Oriani', size = 9, color='#a3a3a3')
    plt.show()

# Reference
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html