# A simple XOR Classifier using Multi-Layer Perceptron Neural Network

import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# define the input set and expected output
input_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_set = [0, 1, 1, 0]

# create the classifier based on Multi-Layer perceptron with the following arguments
model = nn.MLPRegressor(activation='tanh', max_iter=5000, hidden_layer_sizes=(5,))

# train the classifier
model = model.fit(input_set, target_set)

# results
score = model.score(input_set, target_set)
pred = model.predict(input_set)
mse = metrics.mean_squared_error(target_set, pred)

# print some metrics
print('score:', score)
print('predictions: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*pred))
print('expected:', target_set)
print('mean squared error: {:.4f}'.format(mse))

# some asserts for the trained model
assert model.predict([[0, 0]]) < [0.5]
assert model.predict([[0, 1]]) > [0.5]
assert model.predict([[1, 0]]) > [0.5]
assert model.predict([[1, 1]]) < [0.5]

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
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html