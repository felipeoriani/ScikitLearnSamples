import sklearn.neural_network as nn
import sklearn.model_selection as ml
import sklearn.datasets as ds
import sklearn.metrics as metrics
import numpy as np
import pandas as pd

def train_iris(model):
    # Read data from iris.csv file.
    # iris_path = os.path.join(os.path.dirname(__file__), '..\\data\\iris.csv')
    # iris_data = pd.read_csv(iris_path)

    # Load the iris dataset (Bunch) from the 'sklearn.datasets' package.
    iris = ds.load_iris()

    # Define a pandas.dataFrame for feature data using the iris dataset.
    iris_data = pd.DataFrame(data = iris.data, columns = iris.feature_names)

    # Generate a new pandas.dataFrame normalized between 0 and 1.
    iris_normalized = iris_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Define a new pandas.dataFrame for the target iris species based on iris dataset, 
    # defining a single column called 'species'. All the data here is based on 
    # 0, 1 and 2 for each class 'setosa', 'versicolor', 'virginica' respectively.
    iris_target = pd.DataFrame(data = iris.target, columns = ['species'])

    # Concat the iris feature and target data into a new dataFrame.
    iris_set = pd.concat([iris_normalized, iris_target], axis=1)

    # Split the iris data set in 70% for training and 30% for testing.
    input_set, test_set = ml.train_test_split(iris_set, test_size = 0.3)

    # Define the input and target dataframes for training set.
    train_input_set = input_set[iris.feature_names]
    train_target_set = input_set.species

    # Define the input and target dataframes for test set,
    test_input_set = test_set[iris.feature_names]
    test_target_set = test_set.species

    # Train the model with the train sets.
    model.fit(train_input_set, train_target_set)

    # Print some results
    print('Model name: ', type(model))

    # Get some stats from the model.

    # Training Set ------------------------------------------------
    train_score = model.score(train_input_set, train_target_set)
    train_predictions = model.predict(train_input_set)
    train_mse = metrics.mean_squared_error(train_target_set, train_predictions)

    print('Results for Training set (overfitting is expected)')
    print(' Train Score:', train_score)    
    print(' Mean Squared Error: {:f}'.format(train_mse))

    # Test Set ----------------------------------------------------
    test_score = model.score(test_input_set, test_target_set)    
    test_predictions = model.predict(test_input_set)
    test_mse = metrics.mean_squared_error(test_target_set, test_predictions)

    print('Results for Test set')
    print(' Test Score:', test_score)
    print(' Mean Squared Error: {:f}'.format(test_mse))

    # Define 2 new dataFrames for test target values and predictions made for it.
    test_set_target = pd.DataFrame(test_target_set.values, columns=['target'])
    test_target_pred = pd.DataFrame(test_predictions, columns=['prediction'])

    # Concat the dataFrames into a new dataFrame.
    result = pd.concat([test_set_target, test_target_pred], axis=1)

    # Define a new column called 'status' with 'Ok' for right predictions made.
    status = list(map(lambda r: "Ok" if r.target == r.prediction else "-", result.itertuples()))
    result.insert(2, 'status', status, True)

    print('Test predictions:')
    print(result)

mlp = nn.MLPClassifier(hidden_layer_sizes=(3, 3), random_state=1, solver='lbfgs')
train_iris(mlp)

# References

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://www.kaggle.com/avk256/iris-with-mlpclassifier
# https://datatofish.com/create-pandas-dataframe/