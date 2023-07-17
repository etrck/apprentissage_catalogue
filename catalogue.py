import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import numpy as np
import pandas as pd

#assert hasattr(tf, "function") # Be sure to use tensorflow 2.0

column_names = ['CRMID', 'RangAll', 'cat_recu', 'commande']
column_names_output = ['CRMID', 'RangAll', 'cat_recu', 'commande', 'cat_recu_predict', 'commande_predict']
dtypes = {'CRMID': float, 'RangAll': float, 'cat_recu': float, 'commande': float}

vbatch_size = 128
vstride = 34
vlength = 33

df_cat_train = pd.read_csv("C:/Users/TRINCKLIN/Documents/AFM/Python/Tneuveu/rnn_dataset/cat_train10.csv", header=None, names=column_names, sep=';', skiprows=1, dtype=dtypes)
df_cat_test = pd.read_csv("C:/Users/TRINCKLIN/Documents/AFM/Python/Tneuveu/rnn_dataset/cat_test10.csv", header=None, names=column_names, sep=';', skiprows=1, dtype=dtypes)
#print(len(df_cat_train))
#print(df_cat_train[:10])

dataset_train_input = np.array(df_cat_train)
dataset_train_output = dataset_train_input[:, 2:]

dataset_test_input = np.array(df_cat_test)
dataset_test_output = dataset_test_input[:, 2:]

#dataset_output = dataset_input[:-1, 2:]
#dataset_output = dataset_input[1:, 2:]
#derniere_ligne = np.array([[0, 0]])
#dataset_output = np.append(dataset_output, derniere_ligne, axis = 0)
#dataset_output = np.append(derniere_ligne, dataset_output, axis = 0)
dataset_test = dataset_train_input[:17, :]

print(len(dataset_train_input))
print(dataset_train_input.shape)
#print(dataset_train_input[:20, :])

print(len(dataset_train_output))
print(dataset_train_output.shape)
#print(dataset_train_output[:20])


mean1 = dataset_train_input[:, :1].mean()
std1  = dataset_train_input[:, :1].std()
dataset_train_input[:, :1] = (dataset_train_input[:, :1] - mean1) / std1
mean2 = dataset_train_input[:, 1:2].mean()
std2  = dataset_train_input[:, 1:2].std()
dataset_train_input[:, 1:2] = (dataset_train_input[:, 1:2] - mean2) / std2

mean11 = dataset_test_input[:, :1].mean()
std12  = dataset_test_input[:, :1].std()
dataset_test_input[:, :1] = (dataset_test_input[:, :1] - mean1) / std1
mean21 = dataset_test_input[:, 1:2].mean()
std22  = dataset_test_input[:, 1:2].std()
dataset_test_input[:, 1:2] = (dataset_test_input[:, 1:2] - mean2) / std2


#print('mean1:', mean1)
#print('std1:', std1)
#print('mean2:', mean2)
#print('std2:', std2)

#mean3 = dataset_input[:, 2:4].mean()
#std3  = dataset_input[:, 2:4].std()
#dataset_input[:, 2:4] = (dataset_input[:, 2:4] - mean3) / std3

#mean4 = dataset_output.mean()
#std4  = dataset_output.std()
#dataset_output = (dataset_output - mean4) / std4

#mean5 = dataset_test.mean()
#std5  = dataset_test.std()
#dataset_test = (dataset_test - mean5) / std5

"""
print(len(dataset_input))
print(dataset_input.shape)
print(dataset_input[:20])

print(len(dataset_output))
print(dataset_output.shape)
print(dataset_output[:20])
"""

train_generator = TimeseriesGenerator(dataset_train_input, dataset_train_output, length=vlength, batch_size=vbatch_size, stride=vstride)
test_generator = TimeseriesGenerator(dataset_test_input, dataset_test_output, length=vlength, batch_size=vbatch_size, stride=vstride)

x,y=train_generator[0]
print(f'Number of batch trains available : ', len(train_generator))
print('batch x shape : ',x.shape)
print('batch y shape : ',y.shape)

#print('batch x shape : ',x)
#print('batch y shape : ',y)

model = keras.models.Sequential()
model.add( keras.layers.InputLayer(input_shape=(vlength, 4)))
model.add( keras.layers.LSTM(200, return_sequences=False, activation='relu'))
model.add( keras.layers.Dense(200))
model.add( keras.layers.Dense(2))
model.summary()

model.compile(optimizer='rmsprop', 
              loss='mse', 
              metrics = ['mae'])

history=model.fit(train_generator, epochs = 50, validation_data = test_generator)
                  #verbose = 1,
                  #validation_data = test_generator,
                  #callbacks = [bestmodel_callback])
#
# dénormalisation du dataset d'input
#
output_temp = np.array(df_cat_test)
output = []
#output[:, :1] = (output[:, :1] * std1) + mean1
#output[:, 1:2] = (output[:, 1:2] * std2) + mean2

print(len(test_generator))
for i in range(len(test_generator)):
    x,y=test_generator[i]
    prediction = model.predict(x)
    #print(prediction.shape)
    #print(x.shape)
    #print(prediction)
    for j in range(len(x)):
        #print(x[j])
        #print(prediction[j])
        cli_rang = np.array([output_temp[i*vbatch_size * vstride + j*vstride][0], 100])
        final_predict = np.append(cli_rang, prediction[j])
        #print(final_predict)
        a = i*vbatch_size * vstride + j*vstride
        b = i*vbatch_size * vstride + j*vstride + vlength
        #print ('a, b:',a, b)
        if i == 0 and j == 0:
            output = (np.vstack((output_temp[a:b],final_predict)))
        else:
            output = np.vstack((output,(np.vstack((output_temp[a:b],final_predict)))))

print('---------------------- sortie ----------------------')

"""
print(len(output[:, -2:]))
print(output[:, -2:].shape)
print(len(output_temp))
print(output_temp.shape)
print(output)
#print(output[:, -2:])
"""
result = np.hstack((output_temp, output[:, -2:]))

print('---------------------- result ----------------------')
print(len(result))
print(result.shape)

df = pd.DataFrame(result, columns=column_names_output)
df.to_csv("C:/Users/TRINCKLIN/Documents/AFM/Python/Tneuveu/rnn_dataset/cat_test_result10.csv", index=False, sep=';', header=True)
"""



data_apredire = np.expand_dims(dataset_test, axis=0)
print(data_apredire.shape)
print(data_apredire)
prediction = model.predict(data_apredire)
print(prediction)
"""
#dataset_test[:, 2:4] = (dataset_test[:, 2:4] * std3) / mean3
#print(dataset_test)

#mean4 = prediction.mean()
#std4  = prediction.std()
#prediction = (prediction * std4) + mean4
#print(prediction)