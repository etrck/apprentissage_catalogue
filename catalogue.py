import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
#from sklearn.metrics import precision_score, recall_score

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

import pyodbc

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
MssqlActive1 = ConnectMssql['active']
MssqlActive12 = ConnectMssql['active2']
MssqlConnect1 = ConnectMssql['connect']
#tabInputTrain1 = ConnectMssql['tabInputTrain']
#tabInputVal1 = ConnectMssql['tabInputVal']
#tabInputTest1 = ConnectMssql['tabInputTest']
tabOutputTest1 = ConnectMssql['tabOutputTest']

ConnectPyodbc = config['ConnectPyodbc']
MssqlActive2 = ConnectPyodbc['active']
MssqlConnect2 = ConnectPyodbc['connect']
tabInputTrain2 = ConnectPyodbc['tabInputTrain']
tabInputVal2 = ConnectPyodbc['tabInputVal']
tabInputTest2 = ConnectPyodbc['tabInputTest']
#tabOutputTest2 = ConnectPyodbc['tabOutputTest']

Parametre = config['Parametre']
vbatch_size = Parametre['batch_size']
vstride = Parametre['stride']
vlength = Parametre['length']
vparameter = Parametre['parameter']
vepoch = Parametre['epoch']
vLSTM = Parametre['LSTM']
ventrainement = Parametre['entrainement']

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

if MssqlActive1:
    connConfig = SqlConnection(MssqlConnect1)
    connConfig.Open()

if MssqlActive2:
    connPyodbc = pyodbc.connect(MssqlConnect2)

column_names_output = ['CRMID', 'RangAll', 'is_order', 'is_order_predict']
# pour les fichiers csv
dtypes = {'order_amount': float}
#pour les donnees sql
dtypes2 = {'order_amount': np.float64}
train_input = []
test_input = []

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut lecture des données')

if MssqlActive2:
    if ventrainement:
        cursor = connPyodbc.cursor()
        queryTrain = tabInputTrain2    
        cursor.execute(queryTrain)
        result_rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        train_input = np.array(result_rows)
        df_train = pd.DataFrame(train_input, columns=column_names).astype(dtypes2)
        cursor.close()
        del train_input

        cursor = connPyodbc.cursor()
        queryVal = tabInputVal2    
        cursor.execute(queryVal)
        result_rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        val_input = np.array(result_rows)
        df_val = pd.DataFrame(val_input, columns=column_names).astype(dtypes2)
        cursor.close()
        del val_input

    cursor = connPyodbc.cursor()
    queryTest = tabInputTest2
    cursor.execute(queryTest)
    result_rows = cursor.fetchall()
    column_names2 = [desc[0] for desc in cursor.description]
    test_input = np.array(result_rows)
    df_test = pd.DataFrame(test_input, columns=column_names2).astype(dtypes2)
    cursor.close()
    del result_rows
    del test_input

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'fin lecture des données')

print('len df_train:',len(df_train))
print('len df_val:',len(df_val))
print('len df_test:',len(df_test))
print('echantillon du jeu de test:', df_test[:10])

# 
# contruction des dataset input et output
# input : toutes les données sauf crmid et rangall
# output : uniquement is_order
#
# j'enleve le crmid + rangAll mon dataset d'entrainement d'input
if ventrainement:
    dataset_train_input = np.array(df_train)
    # suppression du DF train
    del df_train
    dataset_train_input = dataset_train_input[:, 2:]
    nb_lignes_train = len(dataset_train_input)
    # je ne garde que le commande de mon dataset d'entrainement d'input
    dataset_train_output = dataset_train_input[:, -1:]

    dataset_val_input = np.array(df_val)
    # suppression du DF val
    del df_val
    dataset_val_input = dataset_val_input[:, 2:]
    nb_lignes_train = len(dataset_val_input)
    # je ne garde que le commande de mon dataset d'entrainement d'input
    dataset_val_output = dataset_val_input[:, -1:]

# j'enleve le crmid + rang All mon dataset de test d'input
dataset_test_input = np.array(df_test)
dataset_test_input = dataset_test_input[:, 2:]
nb_lignes_test = len(dataset_test_input)
# je ne garde que le commande de mon dataset de test d'input
dataset_test_output = dataset_test_input[:, -1:]

#
# normalisation de la donnée order_amount
#
if ventrainement:
    # normalisation du mntOrder du dataset d'entrainement
    mean11 = dataset_train_input[:, 21:22].mean()
    std11 = dataset_train_input[:, 21:22].std()
    dataset_train_input[:, 21:22] = (dataset_train_input[:, 21:22] - mean11) / std11

    # normalisation du mntOrder du dataset de validation à partir des valeurs des données d'entrainement
    dataset_val_input[:, 21:22] = (dataset_val_input[:, 21:22] - mean11) / std11

    # normalisation du mntOrder du dataset de test à partir des valeurs des données d'entrainement
    dataset_test_input[:, 21:22] = (dataset_test_input[:, 21:22] - mean11) / std11

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut generation time series')

#
# generation des times series par batchs (lots) pour les data set train, val et test
#
if ventrainement:
    train_generator = TimeseriesGenerator(dataset_train_input, dataset_train_output, length=vlength, batch_size=vbatch_size, stride=vstride)
    val_generator = TimeseriesGenerator(dataset_val_input, dataset_val_output, length=vlength, batch_size=vbatch_size, stride=vstride)
    # suppression des dataset train input/output
    del dataset_train_input
    del dataset_train_output
    del dataset_val_input
    del dataset_val_output

test_generator = TimeseriesGenerator(dataset_test_input, dataset_test_output, length=vlength, batch_size=vbatch_size, stride=vstride)
# suppression des dataset val/test input/output
del dataset_test_input
del dataset_test_output

if ventrainement:
    x,y=train_generator[0]
    print(f'Number of batch trains available : ', len(train_generator))
    print('batch x shape : ',x.shape)
    print('batch y shape : ',y.shape)

# 13400951
# for j in range(vlength):
#     print('train_generator:', x[0][j][20], x[0][j][25], x[0][j][26], '-', y[0])
# 12472501
# print('------------------------------------')
# for j in range(vlength):
#     print('train_generator:', x[3][j][20], x[3][j][25], x[3][j][26], '-', y[3])
    
x,y=test_generator[0]
print(f'Number of batch trains available : ', len(test_generator))
print('batch x shape : ',x.shape)
print('batch y shape : ',y.shape)
#
# définition du modele d'entrainement LSTM
#
if ventrainement:
    model = keras.models.Sequential()
    model.add( keras.layers.InputLayer(input_shape=(vlength, vparameter)))
    model.add( keras.layers.LSTM(vLSTM, return_sequences=False, activation='relu'))
    #model.add( keras.layers.Dense(200))
    #model.add( keras.layers.Dense(1, activation='sigmoid'))
    model.add( keras.layers.Dense(1))
    model.summary()
#
# paramètre du modele
#
    model.compile(optimizer='rmsprop', loss='mse', metrics = ['mae'])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut entrainement du modele')
#
# entrainement du modele
#
    #class_weight = {0: 0.95, 1: 0.05}
    #history=model.fit(train_generator, epochs = 5, validation_data = test_generator, class_weight=class_weight)
    history=model.fit(train_generator, epochs = vepoch, validation_data = val_generator)
                  #verbose = 1,
                  #validation_data = test_generator,
                  #callbacks = [bestmodel_callback])
    
    del train_generator
    del val_generator

    # Sauvegarder le modèle
    model.save('lstm_catalogue_v1.h5')

if not ventrainement:
    # Charger le modèle
    model = keras.models.load_model('lstm_catalogue_v1.h5')

#
# debut de la prediction du data set de test
#
# on récupère le crmid + rangall
output_temp1 = np.array(df_test)[:, :2]
#output_temp1 = output_temp1[:, :2]

# on récupère le commande
output_temp2 = np.array(df_test)[:, -1:]
#output_temp2 = output_temp2[:, -1:]
del df_test

# fusion des deux tableau --> crmid, rangall, cat_commande
output_temp = np.concatenate((output_temp1, output_temp2), axis=1)
del output_temp1
del output_temp2

output = []

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut prediction')

print(len(test_generator))
for i in range(len(test_generator)):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'prediction n°:', i)
    x,y=test_generator[i]
    prediction = model.predict(x)
    for j in range(len(x)):
        # on recup quelques données liées à la prediction (sequence suivante)
        # recup crmid + rangall (=stride) + is_order réèl de la sequence suivante)
        cli_rang = np.array([output_temp[i*vbatch_size * vstride + j*vstride][0],vstride,output_temp[i*vbatch_size * vstride + j*vstride + vstride - 1][2]])
        # on concatène avec la prediction
        final_predict = np.append(cli_rang, prediction[j])
        # on ajoute chaque prédiction à la sortie
        if i == 0 and j == 0:
            output = final_predict
        else:
            output = (np.vstack((output,final_predict)))
        #a = i*vbatch_size * vstride + j*vstride
        #b = i*vbatch_size * vstride + j*vstride + vlength
        #print ('a, b:',a, b)
        #print('output_temp[a:b]', output_temp[a:b])
        # on concatene pour chaque client (sequence de client) verticalement la sequence de test (vlenght) + la prediction (vlenght + 1)
        #if i == 0 and j == 0:
        #    output = (np.vstack((output_temp[a:b],final_predict)))
        #else:
        #    output = np.vstack((output,(np.vstack((output_temp[a:b],final_predict)))))

print('---------------------- output ----------------------')
print('shape output:', output.shape)
# quelques echantillons de la sortie
for i in range(10):
    print('ouput:', i, output[i])

df = pd.DataFrame(output, columns=column_names_output)
print('output:', df)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut ecriture des données')
if MssqlActive1:
    table_name = tabOutputTest1

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

if MssqlActive2:
    connPyodbc.close()

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'fin traitement')