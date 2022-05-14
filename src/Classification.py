
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris.data
y = iris.target

data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
label = data['target']
data.drop('target', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=np.random.randint(1, 10), test_size=0.2)

mlp = MLPClassifier(random_state=1, max_iter=400)
model = mlp.fit(X_train, y_train)
model.score(X_test, y_test)
prediction = mlp.predict(X_test)

print(mlp.score(X_test, y_test))
