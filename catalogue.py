import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import numpy as np
import pandas as pd
import pydatatable as dt
import json
import os
import sys
import datetime
import traceback

import clr
clr.AddReference('System.Data')
from System.Data import SqlDbType, DataTable, DataColumn, DataRow
from System.Data.SqlClient import SqlConnection, SqlCommand, SqlParameter, SqlBulkCopy

def convert_to_datatable(df):
    dt = DataTable()

    # Adding columns to the DataTable
    for col in df.columns:
        dtype = df.dtypes[col]
        data_type = None

        if dtype == "int64":
            data_type = int
        elif dtype == "float64":
            data_type = float
        elif dtype == "object":
            data_type = str
        # Add more data type mappings for other data types as needed

        if data_type:
            dt.Columns.Add(DataColumn(col, data_type))
        else:
            # Default to string data type if the data type is not recognized
            dt.Columns.Add(DataColumn(col, str))

    # Adding rows to the DataTable
    for row in df.itertuples(index=False):
        #print("dt:", dt, row)        
        dt_row = dt.NewRow()
        for i, col in enumerate(df.columns):
            dt_row[i] = getattr(row, col)
        dt.Rows.Add(dt_row)

    return dt

# Variable pour stocker la sortie d'erreur
error_message = ""

# Chemin d'accès du fichier exécutable
exe_path = os.path.abspath(sys.argv[0])

# Répertoire parent du fichier exécutable
script_dir = os.path.dirname(exe_path)

file_path = os.path.join(script_dir, 'config.json')
with open(file_path) as json_file:
    config = json.load(json_file)

ConnectMssql = config['ConnectMssql']
MssqlActive = ConnectMssql['active']
MssqlConnect = ConnectMssql['connect']
tabInputTrain = ConnectMssql['tabInputTrain']
tabInputTest = ConnectMssql['tabInputTest']
tabOutputTest = ConnectMssql['tabOutputTest']

Parametre = config['Parametre']
vbatch_size = Parametre['batch_size']
vstride = Parametre['stride']
vlength = Parametre['length']

FichierCsv = config['FichierCsv']
csvActive = FichierCsv['active']
InputTrain = FichierCsv['InputTrain']
InputTest = FichierCsv['InputTest']
OutPutTest = FichierCsv['OutPutTest']
out = OutPutTest.replace('[]', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3])

Logs = config['Logs']
logActive = Logs['active']
log = Logs['log']

# définie le fichier de log pour trace des différentes actions du pgm
if logActive:
    file_log = os.path.join(log, 'Catalogue' +  '_' + datetime.datetime.now().strftime("%Y-%m-%d") + '.log')
    sys.stdout = open(file_log, "a")

print(' ')
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'début traitement')

if MssqlActive:
    connConfig = SqlConnection(MssqlConnect)
    connConfig.Open()

#assert hasattr(tf, "function") # Be sure to use tensorflow 2.0

column_names = ['CRMID', 'RangAll', 'RangCat', 'cat_recu', 'commande']
column_names_output = ['CRMID', 'RangAll', 'RangCat', 'cat_recu', 'commande', 'commande_predict']
# pour les fichiers csv
dtypes = {'CRMID': float, 'RangAll': float, 'RangCat': float, 'cat_recu': float, 'commande': float}
#pour les donnees sql
dtypes2 = {'CRMID': np.float64, 'RangAll': np.float64, 'RangCat': np.float64, 'cat_recu': np.float64, 'commande': np.float64}
train_input = []
test_input = []

#vbatch_size = 128
#vstride = 34
#vlength = 33

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut lecture des données')
if csvActive:
    fichiers = [fichier for fichier in os.listdir(InputTrain)]
    for fichier in fichiers:
        fichier_en_cours = fichier
        df_fichier = pd.read_csv(os.path.join(InputTrain, fichier), header=None, names=column_names, sep=';', skiprows=1, encoding='ISO-8859-1', dtype=dtypes)
        if len(df_fichier) != 0:
            train_input.append(df_fichier)
    df_train = pd.concat(train_input, axis=0, ignore_index=True)

    fichiers = [fichier for fichier in os.listdir(InputTest)]
    for fichier in fichiers:
        fichier_en_cours = fichier
        df_fichier = pd.read_csv(os.path.join(InputTest, fichier), header=None, names=column_names, sep=';', skiprows=1, encoding='ISO-8859-1', dtype=dtypes)
        if len(df_fichier) != 0:
            test_input.append(df_fichier)
    nb_lignes_test = len(test_input)
    df_test = pd.concat(test_input, axis=0, ignore_index=True)

if MssqlActive:
    #queryTrain = "SELECT TOP (340) [CRMID] ,[RangAll] ,[cat_recu] ,[commande] FROM [David].[dbo].[cat_test10] order by CRMID, RangAll {}".format(tabInputTrain)
    queryTrain = tabInputTrain
    command = SqlCommand(queryTrain, connConfig)
    reader = command.ExecuteReader()
    while reader.Read():
        col_crmid = reader["CRMID"]
        col_RangAll = reader["RangAll"]
        col_RangCat = reader["RangCat"]
        col_cat_recu = reader["cat_recu"]
        col_commande = reader["commande"]
        train_input.append((col_crmid, col_RangAll, col_RangCat, col_cat_recu, col_commande))
    reader.Close()
    df_train = pd.DataFrame(train_input, columns=column_names).astype(dtypes2)

    queryTest = tabInputTest
    command = SqlCommand(queryTest, connConfig)
    reader = command.ExecuteReader()
    while reader.Read():
        col_crmid = reader["CRMID"]
        col_RangAll = reader["RangAll"]
        col_RangCat = reader["RangCat"]
        col_cat_recu = reader["cat_recu"]
        col_commande = reader["commande"]
        test_input.append((col_crmid, col_RangAll, col_RangCat, col_cat_recu, col_commande))
    reader.Close()
    df_test = pd.DataFrame(test_input, columns=column_names).astype(dtypes2)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'fin lecture des données')

#df_cat_train = pd.read_csv("C:/Users/TRINCKLIN/Documents/AFM/Python/github/apprentissage_catalogue/in/cat_train10.csv", header=None, names=column_names, sep=';', skiprows=1, dtype=dtypes)
#df_cat_test = pd.read_csv("C:/Users/TRINCKLIN/Documents/AFM/Python/github/apprentissage_catalogue/in/cat_test10.csv", header=None, names=column_names, sep=';', skiprows=1, dtype=dtypes)
#print(len(df_cat_train))
#print(df_cat_train[:10])

# j'enleve le crmid mon dateset d'entrainement d'input
dataset_train_input = np.array(df_train)
dataset_train_input = dataset_train_input[:, 1:]
nb_lignes_train = len(dataset_train_input)
# je ne garde que le commande de mon dataset d'entrainement d'input
dataset_train_output = dataset_train_input[:, 3:]

# j'enleve le crmid mon dateset de test d'input
dataset_test_input = np.array(df_test)
dataset_test_input = dataset_test_input[:, 1:]
nb_lignes_test = len(dataset_test_input)
# je ne garde que le commande de mon dataset de test d'input
dataset_test_output = dataset_test_input[:, 3:]

#dataset_output = dataset_input[:-1, 2:]
#dataset_output = dataset_input[1:, 2:]
#derniere_ligne = np.array([[0, 0]])
#dataset_output = np.append(dataset_output, derniere_ligne, axis = 0)
#dataset_output = np.append(derniere_ligne, dataset_output, axis = 0)
#dataset_test = dataset_train_input[:17, :]

print(len(dataset_train_input))
print(dataset_train_input.shape)
#print(dataset_train_input[:20, :])

print(len(dataset_train_output))
print(dataset_train_output.shape)
#print(dataset_train_output[:20])

# normalisation su crmid du dataset d'entrainement
#mean1 = dataset_train_input[:, :1].mean()
#std1  = dataset_train_input[:, :1].std()
#dataset_train_input[:, :1] = (dataset_train_input[:, :1] - mean1) / std1

# normalisation su crmid du dataset de test
#mean11 = dataset_test_input[:, :1].mean()
#std12  = dataset_test_input[:, :1].std()
#dataset_test_input[:, :1] = (dataset_test_input[:, :1] - mean1) / std1

# normalisation su RangAll du dataset d'entrainement
mean2 = dataset_train_input[:, :1].mean()
std2  = dataset_train_input[:, :1].std()
dataset_train_input[:, :1] = (dataset_train_input[:, :1] - mean2) / std2

# normalisation su RangAll du dataset de test
mean21 = dataset_test_input[:, :1].mean()
std22  = dataset_test_input[:, :1].std()
dataset_test_input[:, :1] = (dataset_test_input[:, :1] - mean2) / std2

# normalisation su RangCat du dataset d'entrainement
mean3 = dataset_train_input[:, 1:2].mean()
std3  = dataset_train_input[:, 1:2].std()
dataset_train_input[:, 1:2] = (dataset_train_input[:, 1:2] - mean3) / std3

# normalisation su RangCat du dataset de test
mean31 = dataset_test_input[:, 1:2].mean()
std32  = dataset_test_input[:, 1:2].std()
dataset_test_input[:, 1:2] = (dataset_test_input[:, 1:2] - mean3) / std3

print('mean2:', mean2)
print('std2:', std2)
print('mean2:', mean3)
print('std2:', std3)

#mean3 = dataset_input[:, 2:4].mean()
#std3  = dataset_input[:, 2:4].std()
#dataset_input[:, 2:4] = (dataset_input[:, 2:4] - mean3) / std3

#mean4 = dataset_output.mean()
#std4  = dataset_output.std()
#dataset_output = (dataset_output - mean4) / std4

#mean5 = dataset_test.mean()
#std5  = dataset_test.std()
#dataset_test = (dataset_test - mean5) / std5

#print(len(dataset_input))
#print(dataset_input.shape)
#print(dataset_input[:20])

#print(len(dataset_output))
#print(dataset_output.shape)
#print(dataset_output[:20])

print(vbatch_size)
print(vstride)
print(vlength)

print(nb_lignes_train)
print(nb_lignes_test)

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
#model.add( keras.layers.Dense(200))
model.add( keras.layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop', 
              loss='mse', 
              metrics = ['mae'])

history=model.fit(train_generator, epochs = 20, validation_data = test_generator)
                  #verbose = 1,
                  #validation_data = test_generator,
                  #callbacks = [bestmodel_callback])
#
# dénormalisation du dataset d'input
#
output_temp = np.array(df_test)
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
        cli_rang = np.array([output_temp[i*vbatch_size * vstride + j*vstride][0], 100, 100, 1])
        final_predict = np.append(cli_rang, prediction[j])
        #print(final_predict)
        a = i*vbatch_size * vstride + j*vstride
        b = i*vbatch_size * vstride + j*vstride + vlength
        #print ('a, b:',a, b)
        #print(output_temp[a:b])
        #print(final_predict)
        #print(cli_rang)       
        if i == 0 and j == 0:
            output = (np.vstack((output_temp[a:b],final_predict)))
        else:
            output = np.vstack((output,(np.vstack((output_temp[a:b],final_predict)))))

print('---------------------- sortie ----------------------')

#print(len(output[:, -2:]))
#print(output[:, -2:].shape)
#print(len(output_temp))
#print(output_temp.shape)
#print(output.shape)
#print(output)
#print(output[:, -1:])

result = np.hstack((output_temp, output[:, -1:]))

print('---------------------- result ----------------------')
print(len(result))
print(result.shape)
#print(result)

df = pd.DataFrame(result, columns=column_names_output)
#print(df)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut ecriture des données')
if csvActive:
    df.to_csv(out, index=False, sep=';', header=True)
if MssqlActive:
    table_name = "cat_out11"

    # Convert DataFrame to DataTable using pydatatable
    #dt_table = dt.Frame(df)
    dt_table = convert_to_datatable(df)

    # Configuration de SqlBulkCopy
    bulk_copy = SqlBulkCopy(connConfig)
    bulk_copy.DestinationTableName = table_name

    # Exécution du Bulk Copy
    try:
        bulk_copy.WriteToServer(dt_table)
    finally:
        bulk_copy.Close()    
        # Fermeture de la connexion
    connConfig.Close()
    #df.to_sql(tabOutputTest, connConfig, if_exists="replace", index=False)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'fin ecriture des données')

#data_apredire = np.expand_dims(dataset_test, axis=0)
#print(data_apredire.shape)
#print(data_apredire)
#prediction = model.predict(data_apredire)
#print(prediction)

#dataset_test[:, 2:4] = (dataset_test[:, 2:4] * std3) / mean3
#print(dataset_test)

#mean4 = prediction.mean()
#std4  = prediction.std()
#prediction = (prediction * std4) + mean4
#print(prediction)

#connConfig.Close()
