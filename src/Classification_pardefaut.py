from sklearn import datasets
import numpy as np
import pandas as pd
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

mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                    batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200,
                    shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                    nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
model = mlp.fit(X_train, y_train)
model.score(X_test, y_test)
prediction = mlp.predict(X_test)

print(mlp.score(X_test, y_test))
