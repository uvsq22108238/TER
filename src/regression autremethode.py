
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
diabetes_X, diabetes_Y = datasets.load_diabetes(return_X_y=True)

# ADD COLUMNS NAMES
df1 = pd.DataFrame(diabetes_X, columns=["age","sex","bmi","bp", "tc", "ldl", "hdl","tch", "ltg", "glu"])

df2 = pd.DataFrame(diabetes_Y, columns=["disease_progression"])
# MARGE IN ONE TABLE
df = pd.merge(df1,df2, left_index=True, right_index=True)
# Check for null values
df.isnull().sum()
# affectations, & drop from X our quantitative measure of disease progression
diabetes_Y = df.disease_progression
diabetes_X = df.drop(['disease_progression'], axis=1)



# Split into validation and training data
diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=0.2, random_state=1)


# MlP Training
mlp = MLPRegressor(hidden_layer_sizes=(6, 6),
                   max_iter=1000, learning_rate_init=0.07)

model = mlp.fit(diabetes_X_train,diabetes_Y_train)

print("Training set score: %f" % model.score(diabetes_X_train,diabetes_Y_train))
print("Test set score: %f" % model.score(diabetes_X_test, diabetes_Y_test))

#Prediction
diabetes_predicion = model.predict(diabetes_X_test)

# Cross validation
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(model, diabetes_X_train,diabetes_Y_train, cv=5, scoring='r2')
#print('Mean of Cross validation score: ',scores.mean())

# Use only one feature to predict  make an evenly spaced array to avoid size pb
diabetes_X_test =np.arange(0,len(diabetes_X_test),1)

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_Y_test, color="black")
plt.plot(diabetes_X_test, diabetes_predicion, color="blue")
plt.show()