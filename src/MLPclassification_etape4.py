import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report


#charging the dateset result
#df = pd.read_csv('breastCancer.csv')
dataframe = pd.read_csv('C:/eRisk2020_T2_TRAINING_DATA/test/breastCancer.csv')

#showing the seven first rows
dataframe.head(7)
dataframe.isna().sum()
dataframe.info()
dataframe.describe()
# i found 16 rows of "?" on the dataset thats why i used this function to fix it
dataset = dataframe.values
dataset = dataset.astype(str)

def taking_off(data):
    dataset_2 = []
    for row in data:
        if "?" not in row:
            dataset_2.append(row)
    return dataset_2
#  so here at first Malignant was a 4 then turned it into a 1  and Benign was a 2 and i also turned it into a 0
# because its easier to work with binary values

def to_zero_andone(data):
    for idx, val in enumerate(data):
        if str(val) == "2":
            output_x[idx] = 0
        elif str(val) == "4":
            output_x[idx] = 1
    return data

#define a 2D array
dataset = np.array(taking_off(dataset))

# here we uploaded every column but the first "id" (cz we wont be needing it to calculate) and the last "class"
input_x = dataset[:, 1:-1]

output_x = dataset[:, -1]

output_x = to_zero_andone(output_x) #binarisation

# here we use train_test_split to split the dataset to two parts 75 % for the train and 25% for the test
validation_size = 0.25
train_x, validation_x, train_y, validation_y = train_test_split(input_x, output_x, test_size=validation_size)

# here i used MLP you may ask why well because its suitable for classification prediction problems

# the Keras library is a model and The simplest model is defined in the Sequentialclass which is a linear stack of Layers

layer_1 = 10

# ReLU is the most used activation function in the world right now;
# its a linear function that will output the input directly if is positive, otherwise, it will output zero.
function = 'relu'

# here i created Activation object and add it directly to the model
# There are a large number of core Layer types for standard neural networks.
# here i used Dense which means Fully connected layer and the most common type of layer used on multi-layer perceptron models.
# and also Dropout which means setting a fraction of inputs to zero in an effort to reduce over fitting.

model = Sequential()
model.add(Dense(layer_1, activation=function))
model.add(Dropout(0.2))
model.add(Dense(layer_1, activation=function))
model.add(Dropout(0.2))
model.add(Dense(layer_1, activation=function))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# so my model here is defined  now, it needs to be compiled.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# here The model is trained on NumPy arrays using the fit() function
history = model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=5, batch_size=4,
                    shuffle=True, verbose=1)

# now my model is trained,we can use it to make predictions on test data
# here i used evaluate() To calculate the loss values for input data.
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# so here it shows that we are going to test 171 sample

# now here i tested the 171 samples to see in which class it actually belongs
target_names = ['class 0', 'class 1']
y_predicted = model.predict(validation_x, batch_size=4, verbose=1)
i = 0
for row in y_predicted:
    print(row)
    if row[0] > 0.5:
        y_predicted[i] = 1
    else:
        y_predicted[i] = 0
    i += 1

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

print(classification_report(validation_y.astype(int), y_predicted.astype(int)))
predict_this = np.array([[1,1,1,1,10,1,1,1,1],[8,10,10,8,7,10,9,7,1]])
this_is = model.predict(predict_this)

