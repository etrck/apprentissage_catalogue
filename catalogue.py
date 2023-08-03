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
tabInputTrain1 = ConnectMssql['tabInputTrain']
tabInputTest1 = ConnectMssql['tabInputTest']
tabOutputTest1 = ConnectMssql['tabOutputTest']

ConnectPyodbc = config['ConnectPyodbc']
MssqlActive2 = ConnectPyodbc['active']
MssqlConnect2 = ConnectPyodbc['connect']
tabInputTrain2 = ConnectPyodbc['tabInputTrain']
tabInputTest2 = ConnectPyodbc['tabInputTest']
tabOutputTest2 = ConnectPyodbc['tabOutputTest']

Parametre = config['Parametre']
vbatch_size = Parametre['batch_size']
vstride = Parametre['stride']
vlength = Parametre['length']
vparameter = Parametre['parameter']

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

#assert hasattr(tf, "function") # Be sure to use tensorflow 2.0

column_names = ['CRMID',
                'RangAll',
                'RangCat1', 'RangCat2', 'RangCat3', 'RangCat4', 'RangCat5', 'RangCat6', 'RangCat7', 'RangCat8', 'RangCat9', 'RangCat10', 'RangCat11', 'RangCat12', 'RangCat13', 'RangCat14', 'RangCat15', 'RangCat16', 'RangCat17',
                'canal1', 'canal2', 'canal12',
                'fam_ac', 'fam_af', 'fam_ag', 'fam_an', 'fam_ar', 'fam_at', 'fam_cb', 'fam_ci', 'fam_cm', 'fam_cq', 'fam_cx', 'fam_hd', 'fam_hg', 'fam_hh', 'fam_hj', 'fam_hk', 'fam_hp', 'fam_hu', 'fam_hv', 'fam_kdo',
                'line_is_valid',
                'order_amount',
                'has_fees',
                'ofr_fid',
                'ofr_rec',
                'catalogog_is_sent',
                'is_order']
column_names_output = ['CRMID', 'RangAll', 'is_order', 'is_order_predict']
# pour les fichiers csv
dtypes = {'order_amount': float}
#pour les donnees sql
dtypes2 = {'order_amount': np.float64}
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

if MssqlActive12:
    queryTrain = tabInputTrain1
    command = SqlCommand(queryTrain, connConfig)
    reader = command.ExecuteReader()
    i = 0
    j = 0
    while reader.Read():
        i = i + 1
        j = j + 1
        if i == 100000:
            i = 0
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'en cours lecture data train :', j)

        col_crmid = reader["CRMID"]
        col_RangAll = reader["RangAll"]
        col_RangCat1 = reader["cat_1"]
        col_RangCat2 = reader["cat_2"]
        col_RangCat3 = reader["cat_3"]
        col_RangCat4 = reader["cat_4"]
        col_RangCat5 = reader["cat_5"]
        col_RangCat6 = reader["cat_6"]
        col_RangCat7 = reader["cat_7"]
        col_RangCat8 = reader["cat_8"]
        col_RangCat9 = reader["cat_9"]
        col_RangCat10 = reader["cat_10"]
        col_RangCat11 = reader["cat_11"]
        col_RangCat12 = reader["cat_12"]
        col_RangCat13 = reader["cat_13"]
        col_RangCat14 = reader["cat_14"]
        col_RangCat15 = reader["cat_15"]
        col_RangCat16 = reader["cat_16"]
        col_RangCat17 = reader["cat_17"]
        col_canal1 = reader["can_1"]
        col_canal2 = reader["can_2"]
        col_canal12 = reader["can_12"]
        col_fam_ac = reader["fam_ac"]
        col_fam_af = reader["fam_af"]
        col_fam_ag = reader["fam_ag"]
        col_fam_an = reader["fam_an"]
        col_fam_ar = reader["fam_ar"]
        col_fam_at = reader["fam_at"]
        col_fam_cb = reader["fam_cb"]
        col_fam_ci = reader["fam_ci"]
        col_fam_cm = reader["fam_cm"]
        col_fam_cq = reader["fam_cq"]
        col_fam_cx = reader["fam_cx"]
        col_fam_hd = reader["fam_hd"]
        col_fam_hg = reader["fam_hg"]
        col_fam_hh = reader["fam_hh"]
        col_fam_hj = reader["fam_hj"]
        col_fam_hk = reader["fam_hk"]
        col_fam_hp = reader["fam_hp"]
        col_fam_hu = reader["fam_hu"]
        col_fam_hv = reader["fam_hv"]
        col_fam_kdo = reader["fam_kdo"]
        col_line_is_valid = reader["line_is_valid"]
        col_order_amount = reader["order_amount"]
        col_has_fees = reader["has_fees"]
        col_ofr_fid = reader["ofr_fid"]
        col_ofr_rec = reader["ofr_rec"]
        col_catalog_is_sent = reader["catalog_is_sent"]
        col_is_order = reader["is_order"]
        train_input.append((col_crmid, col_RangAll,
                            col_RangCat1, col_RangCat2, col_RangCat3, col_RangCat4, col_RangCat5, col_RangCat6, col_RangCat7, col_RangCat8, col_RangCat9, col_RangCat10, col_RangCat11, col_RangCat12, col_RangCat13, col_RangCat14, col_RangCat15, col_RangCat16, col_RangCat17, 
	                        col_canal1, col_canal2, col_canal12, 
	                        col_fam_ac, col_fam_af, col_fam_ag, col_fam_an, col_fam_ar, col_fam_at, col_fam_cb, col_fam_ci, col_fam_cm, col_fam_cq, col_fam_cx, col_fam_hd, col_fam_hg, col_fam_hh, col_fam_hj, col_fam_hk, col_fam_hp, col_fam_hu, col_fam_hv, col_fam_kdo,
	                        col_line_is_valid, col_order_amount, col_has_fees, col_ofr_fid, col_ofr_rec, col_catalog_is_sent, col_is_order))
    reader.Close()
    df_train = pd.DataFrame(train_input, columns=column_names).astype(dtypes2)

    queryTest = tabInputTest1
    command = SqlCommand(queryTest, connConfig)
    reader = command.ExecuteReader()
    i = 0
    j = 0
    while reader.Read():
        i = i + 1
        j = j + 1
        if i == 100000:
            i = 0
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'en cours lecture data test :', j)

        col_crmid = reader["CRMID"]
        col_RangAll = reader["RangAll"]
        col_RangCat1 = reader["cat_1"]
        col_RangCat2 = reader["cat_2"]
        col_RangCat3 = reader["cat_3"]
        col_RangCat4 = reader["cat_4"]
        col_RangCat5 = reader["cat_5"]
        col_RangCat6 = reader["cat_6"]
        col_RangCat7 = reader["cat_7"]
        col_RangCat8 = reader["cat_8"]
        col_RangCat9 = reader["cat_9"]
        col_RangCat10 = reader["cat_10"]
        col_RangCat11 = reader["cat_11"]
        col_RangCat12 = reader["cat_12"]
        col_RangCat13 = reader["cat_13"]
        col_RangCat14 = reader["cat_14"]
        col_RangCat15 = reader["cat_15"]
        col_RangCat16 = reader["cat_16"]
        col_RangCat17 = reader["cat_17"]
        col_canal1 = reader["can_1"]
        col_canal2 = reader["can_2"]
        col_canal12 = reader["can_12"]
        col_fam_ac = reader["fam_ac"]
        col_fam_af = reader["fam_af"]
        col_fam_ag = reader["fam_ag"]
        col_fam_an = reader["fam_an"]
        col_fam_ar = reader["fam_ar"]
        col_fam_at = reader["fam_at"]
        col_fam_cb = reader["fam_cb"]
        col_fam_ci = reader["fam_ci"]
        col_fam_cm = reader["fam_cm"]
        col_fam_cq = reader["fam_cq"]
        col_fam_cx = reader["fam_cx"]
        col_fam_hd = reader["fam_hd"]
        col_fam_hg = reader["fam_hg"]
        col_fam_hh = reader["fam_hh"]
        col_fam_hj = reader["fam_hj"]
        col_fam_hk = reader["fam_hk"]
        col_fam_hp = reader["fam_hp"]
        col_fam_hu = reader["fam_hu"]
        col_fam_hv = reader["fam_hv"]
        col_fam_kdo = reader["fam_kdo"]
        col_line_is_valid = reader["line_is_valid"]
        col_order_amount = reader["order_amount"]
        col_has_fees = reader["has_fees"]
        col_ofr_fid = reader["ofr_fid"]
        col_ofr_rec = reader["ofr_rec"]
        col_catalog_is_sent = reader["catalog_is_sent"]
        col_is_order = reader["is_order"]
        test_input.append((col_crmid, col_RangAll,
                            col_RangCat1, col_RangCat2, col_RangCat3, col_RangCat4, col_RangCat5, col_RangCat6, col_RangCat7, col_RangCat8, col_RangCat9, col_RangCat10, col_RangCat11, col_RangCat12, col_RangCat13, col_RangCat14, col_RangCat15, col_RangCat16, col_RangCat17, 
	                        col_canal1, col_canal2, col_canal12, 
	                        col_fam_ac, col_fam_af, col_fam_ag, col_fam_an, col_fam_ar, col_fam_at, col_fam_cb, col_fam_ci, col_fam_cm, col_fam_cq, col_fam_cx, col_fam_hd, col_fam_hg, col_fam_hh, col_fam_hj, col_fam_hk, col_fam_hp, col_fam_hu, col_fam_hv, col_fam_kdo,
	                        col_line_is_valid, col_order_amount, col_has_fees, col_ofr_fid, col_ofr_rec, col_catalog_is_sent, col_is_order))
    reader.Close()
    df_test = pd.DataFrame(test_input, columns=column_names).astype(dtypes2)

if MssqlActive2:
    cursor = connPyodbc.cursor()
    queryTrain = tabInputTrain2    
    cursor.execute(queryTrain)
    result_rows = cursor.fetchall()
    column_names2 = [desc[0] for desc in cursor.description]
    train_input = np.array(result_rows)
    df_train = pd.DataFrame(train_input, columns=column_names2).astype(dtypes2)
    cursor.close()

    cursor = connPyodbc.cursor()
    queryTest = tabInputTest2    
    cursor.execute(queryTest)
    result_rows = cursor.fetchall()
    column_names2 = [desc[0] for desc in cursor.description]
    test_input = np.array(result_rows)
    df_test = pd.DataFrame(test_input, columns=column_names2).astype(dtypes2)
    cursor.close()

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'fin lecture des données')

#df_cat_train = pd.read_csv("C:/Users/TRINCKLIN/Documents/AFM/Python/github/apprentissage_catalogue/in/cat_train10.csv", header=None, names=column_names, sep=';', skiprows=1, dtype=dtypes)
#df_cat_test = pd.read_csv("C:/Users/TRINCKLIN/Documents/AFM/Python/github/apprentissage_catalogue/in/cat_test10.csv", header=None, names=column_names, sep=';', skiprows=1, dtype=dtypes)
print(len(df_train))
print(df_train[:10])

print(len(df_test))
print(df_test[:10])
# j'enleve le crmid + rangAll mon dataset d'entrainement d'input
dataset_train_input = np.array(df_train)
dataset_train_input = dataset_train_input[:, 2:]
nb_lignes_train = len(dataset_train_input)
# je ne garde que le commande de mon dataset d'entrainement d'input
dataset_train_output = dataset_train_input[:, -1:]

# j'enleve le crmid + rang All mon dataset de test d'input
dataset_test_input = np.array(df_test)
dataset_test_input = dataset_test_input[:, 2:]
nb_lignes_test = len(dataset_test_input)
# je ne garde que le commande de mon dataset de test d'input
dataset_test_output = dataset_test_input[:, -1:]

#dataset_output = dataset_input[:-1, 2:]
#dataset_output = dataset_input[1:, 2:]
#derniere_ligne = np.array([[0, 0]])
#dataset_output = np.append(dataset_output, derniere_ligne, axis = 0)
#dataset_output = np.append(derniere_ligne, dataset_output, axis = 0)
#dataset_test = dataset_train_input[:17, :]

#print('len train_input', len(dataset_train_input))
#print('shape train_input', dataset_train_input.shape)
#print('train_input', dataset_train_input[:20, :])
#print('train_input order_amount', dataset_train_input[:20, 41:42])

#print(len(dataset_train_output))
#print(dataset_train_output.shape)
#print(dataset_train_output[:20])
#print('test_input order_amount', dataset_test_input[:20, 41:42])

# normalisation du mntOrder du dataset d'entrainement
mean11 = dataset_train_input[:, 41:42].mean()
std11 = dataset_train_input[:, 41:42].std()
dataset_train_input[:, 41:42] = (dataset_train_input[:, 41:42] - mean11) / std11

# normalisation su mntOrder du dataset de test
mean12 = dataset_test_input[:, 41:42].mean()
std12  = dataset_test_input[:, 41:42].std()
dataset_test_input[:, 41:42] = (dataset_test_input[:, 41:42] - mean11) / std11

print('mean2:', mean11)
print('std2:', std11)
print('mean2:', mean12)
print('std2:', std12)

print(vbatch_size)
print(vstride)
print(vlength)

print(nb_lignes_train)
print(nb_lignes_test)

#print(len(dataset_train_input))
#print(dataset_train_input.shape)
#print(dataset_train_input[:20, :])

#print(len(dataset_train_output))
#print(dataset_train_output.shape)
#print(dataset_train_output[:20])

# pad_sequences
# .....

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut generation time series')

train_generator = TimeseriesGenerator(dataset_train_input, dataset_train_output, length=vlength, batch_size=vbatch_size, stride=vstride)
test_generator = TimeseriesGenerator(dataset_test_input, dataset_test_output, length=vlength, batch_size=vbatch_size, stride=vstride)

x,y=train_generator[0]
print(f'Number of batch trains available : ', len(train_generator))
print('batch x shape : ',x.shape)
print('batch y shape : ',y.shape)

x,y=test_generator[0]
print(f'Number of batch trains available : ', len(test_generator))
print('batch x shape : ',x.shape)
print('batch y shape : ',y.shape)
#print('batch x shape : ',x)
#print('batch y shape : ',y)

model = keras.models.Sequential()
model.add( keras.layers.InputLayer(input_shape=(vlength, vparameter)))
model.add( keras.layers.LSTM(50, return_sequences=False, activation='relu'))
#model.add( keras.layers.Dense(200))
model.add( keras.layers.Dense(1, activation='relu'))
model.summary()

model.compile(optimizer='rmsprop', 
              loss='mse', 
              metrics = ['mae'])

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut entrainement du modele')

history=model.fit(train_generator, epochs = 5, validation_data = test_generator)
                  #verbose = 1,
                  #validation_data = test_generator,
                  #callbacks = [bestmodel_callback])
#
# dénormalisation du dataset d'input
#
#
#
#
# on récupère le crmid + rangall
output_temp1 = np.array(df_test)
output_temp1 = output_temp1[:, :2]

# on récupère le commande
output_temp2 = np.array(df_test)
output_temp2 = output_temp2[:, -1:]

print('------------------ output_temp1 --------------------')
print(len(output_temp1))
print(output_temp1.shape)
#print(output_temp1[:20, :])
print('------------------ output_temp2 --------------------')
print(len(output_temp2))
print(output_temp2.shape)
#print(output_temp2[:20, :])

# fusion des deux tableau --> crmid, rangall, cat_commande
output_temp = np.concatenate((output_temp1, output_temp2), axis=1)

print('------------------ output_temp --------------------')
print(len(output_temp))
print(output_temp.shape)
#print(output_temp[:20, :])

output = []
#output[:, :1] = (output[:, :1] * std1) + mean1
#output[:, 1:2] = (output[:, 1:2] * std2) + mean2

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut prediction')

print(len(test_generator))
for i in range(len(test_generator)):
    #print('i:', i)
    x,y=test_generator[i]
    prediction = model.predict(x)
    #print(prediction.shape)
    #print(x.shape)
    #print(prediction)
    for j in range(len(x)):
        #print('j:', j)
        #print(x[j])
        #print(prediction[j])
        # on récupère le crm_id de la prédiction (cli_rang)
        cli_rang = np.array([output_temp[i*vbatch_size * vstride + j*vstride][0],vstride])
        #print('cli_rang:', cli_rang)
        final_predict = np.append(cli_rang, prediction[j])
        #print('final_predict:', final_predict)
        a = i*vbatch_size * vstride + j*vstride
        b = i*vbatch_size * vstride + j*vstride + vlength
        #print ('a, b:',a, b)
        #print('output_temp[a:b]', output_temp[a:b])
        # on concatene pour chaque client (sequence de client) verticalement la sequence de test (vlenght) + la prediction (vlenght + 1)
        if i == 0 and j == 0:
            output = (np.vstack((output_temp[a:b],final_predict)))
        else:
            output = np.vstack((output,(np.vstack((output_temp[a:b],final_predict)))))

print('---------------------- output ----------------------')

#print(len(output[:, -2:]))
#print(output[:, -2:].shape)
#print(len(output_temp))
#print(output_temp.shape)
print(output.shape)
#print(output)
#print(output[:, -1:])

# resultat final : on garde le reel (crmid, rangall, commande) et on ajoute pour le dernier rang la prediction et pour le reste le commande
result = np.hstack((output_temp, output[:, -1:]))
result_predict = result[result[:, 1] == vstride]

print('---------------------- result ----------------------')
print(len(result))
print(result.shape)
#print(result)

print('---------------------- result predict----------------------')
print(len(result_predict))
print(result_predict.shape)
#print(result_predict)

df = pd.DataFrame(result_predict, columns=column_names_output)
print('output:', df)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], 'debut ecriture des données')
if csvActive:
    df.to_csv(out, index=False, sep=';', header=True)

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

if MssqlActive2:
    connPyodbc.close()

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
