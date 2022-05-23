seed=7
import pandas as pd
#charging the dateset result
#df = pd.read_csv('breastCancer.csv')
dataframe = pd.read_csv('C:/eRisk2020_T2_TRAINING_DATA/test/data.csv')
dataframe.head(7)
cols_at_end = ['fractal_dimension_worst', 'diagnosis']
dataframe= dataframe[[c for c in dataframe if c not in cols_at_end]
        + [c for c in cols_at_end if c in dataframe]]
dataframe.drop('Unnamed: 32', inplace=True, axis=1)
dataset = dataframe.values
dataset = dataset.astype(str)

def to_zero_andone(data):
    for idx, val in enumerate(data):
        if str(val) == "M":
            output_x[idx] = 1
        elif str(val) == "B":
            output_x[idx] = 0
    return data

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(150, 100, 50),
        learning_rate_init= 0.007, max_iter=1000)


# here we uploaded every column but the first "id" (cz we wont be needing it to calculate) and the second one
input_x = dataset[:,1:-1]

output_x = dataset[:, -1]

output_x = to_zero_andone(output_x)
validation_size = 0.25
train_x, validation_x, train_y, validation_y = train_test_split(input_x , output_x, test_size=validation_size)

history= model.fit(train_x, train_y )
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements
y_pred = model.predict(validation_x)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,validation_y)

#Printing the accuracy
print("Accuracy of MLPClassifier : ", accuracy(cm))