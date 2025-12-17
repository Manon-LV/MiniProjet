import numpy as np

# Fichier à vérifier
X = np.load('X_20000.npy')
y = np.load('y_20000.npy')

print(f"Shape X: {X.shape}")
print(f"Shape y: {y.shape}")

# Vérification des doublons sur X (et X,y)
X_flat = X.reshape(X.shape[0], -1)

# Doublons sur X uniquement
_, idx_unique_X = np.unique(X_flat, axis=0, return_index=True)
dup_X = X_flat.shape[0] - len(idx_unique_X)
print(f"Doublons sur X uniquement : {dup_X}")

# Doublons sur (X, y) (vrai doublon complet)
XY = np.concatenate([X_flat, y.reshape(-1, 1)], axis=1)
_, idx_unique_XY = np.unique(XY, axis=0, return_index=True)
dup_XY = XY.shape[0] - len(idx_unique_XY)
print(f"Doublons sur (X, y) : {dup_XY}")

# Statistiques sur y
classes, counts = np.unique(y, return_counts=True)
print("Répartition des classes :")
for c, n in zip(classes, counts):
    print(f"  Classe {c}: {n}")

# Optionnel : indices des doublons
if dup_X > 0:
    print("Indices de doublons X (premiers 10) :", np.where(~np.isin(np.arange(X_flat.shape[0]), idx_unique_X))[0][:10])
if dup_XY > 0:
    print("Indices de doublons (X, y) (premiers 10) :", np.where(~np.isin(np.arange(XY.shape[0]), idx_unique_XY))[0][:10])
