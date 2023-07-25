import numpy as np

# Exemple de tableau numpy à deux entrées
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Suppression de la première colonne
data_sans_premiere_colonne = data[:, 2:]
print(data_sans_premiere_colonne.shape)
print(data_sans_premiere_colonne)