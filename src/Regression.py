import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Split the data into training/testing sets
# Tous sauf les 30 derniers pour le train
# Les 30 derniers pour le test
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

# Split the targets into training/testing sets
# Tous sauf les 30 derniers pour le train
# Les 30 derniers pour le test
diabetes_y_train = diabetes_y[:-30]
diabetes_y_test = diabetes_y[-30:]

# Create linear regression object
model = MLPRegressor(hidden_layer_sizes= (150,100),
        learning_rate_init= 0.07, max_iter=1000)

# Train the model using the training sets
model.fit(diabetes_X_train, diabetes_y_train)

#Test de mes données
diabetes_score = model.score(diabetes_X_test, diabetes_y_test)
print(diabetes_score)
#Prediction
diabetes_predicion = model.predict(diabetes_X_test)
# Use only one feature to print the plot
diabetes_X = diabetes_X[:, np.newaxis, 2]
# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_predicion, color="blue")
plt.show()
