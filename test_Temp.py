import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Données d'entrée (série temporelle avec deux caractéristiques)
serie_temps = np.array([[1, 20],
                        [2, 22],
                        [3, 25],
                        [4, 19],
                        [5, 18]])

# Nombre de pas de temps dans la séquence (vlength)
vlength = 3

# Séparation des données en entrée (X) et sortie (y)
X, y = [], []
for i in range(len(serie_temps) - vlength):
    X.append(serie_temps[i:i+vlength])
    y.append(serie_temps[i+vlength, 1])  # La température à prédire (2ème caractéristique)

X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)
print('X:', X)
print('y:', y)

# Création du modèle LSTM
model = Sequential()
model.add(LSTM(10, input_shape=(vlength, 2), activation='relu'))
model.add(Dense(1, activation='linear'))

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X, y, epochs=10, batch_size=1)

# Maintenant, vous pouvez utiliser le modèle pour faire des prédictions
nouvelles_donnees_entree = np.array([[6, 17], [7, 20], [8, 23]])  # Nouvelle séquence de 3 pas de temps
nouvelles_donnees_entree = nouvelles_donnees_entree.reshape((1, vlength, 2))  # Reshape pour correspondre à l'entrée du modèle
print('nouvelles_donnees_entree:', nouvelles_donnees_entree)
prediction = model.predict(nouvelles_donnees_entree)
print(prediction)
