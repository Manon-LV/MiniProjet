#=================================================================================================================================================================
#trainCNN.py
#=================================================================================================================================================================
# Ce fichier entraîne un modèle de réseau de neurones convolutifs (CNN) pour une politique d'agent dans un environnement GridWorld
# en utilisant un dataset pré-généré d'observations et d'actions optimales.
# Le modèle est entraîné à prédire l'action optimale à partir de l'observation de l'environnement.
# Le dataset est chargé à partir de fichiers numpy, et le modèle est optimisé en utilisant la descente de gradient avec la perte d'entropie croisée.

#=================================================================================================================================================================
# Importations
#=================================================================================================================================================================
import torch
import numpy as np
import CNN
import time
import matplotlib.pyplot as plt
from torch.utils.data import random_split
#=================================================================================================================================================================
# Initialisation des listes pour le suivi des performances
#=================================================================================================================================================================
train_losses = []
train_accuracies = []

test_losses = []
test_accuracies = []

epoch_times = []

start_time = time.time()

#=================================================================================================================================================================
# Chargement du dataset
#=================================================================================================================================================================
X = np.load("X.npy")
y = np.load("y.npy")

# Conversion des données en tenseurs PyTorch
X = torch.tensor(X)
y = torch.tensor(y)

# Création du DataLoader pour le dataset
dataset = torch.utils.data.TensorDataset(X, y)
# Division du dataset en ensembles d'entraînement et de test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
#=================================================================================================================================================================
# Entraînement du modèle CNN
#=================================================================================================================================================================
# Initialisation du modèle, de l'optimiseur et de la fonction de perte
model = CNN.CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
model.train()



# Fonction d'évaluation du modèle
def evaluate(model, test_loader, criterion):
    # Met en mode évaluation
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    # Désactivation du calcul des gradients
    with torch.no_grad():
        # boucle sur les batches de données
        for xb, yb in test_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    # Calcul des métriques
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Boucle d'entraînement
for epoch in range(30):
    #Initialisation des variables pour le suivi des performances de l'époque
    epoch_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    # boucle sur les batches de données
    for xb, yb in train_loader:
        # passage avant, calcul de la perte, rétropropagation et mise à jour des poids
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # mise à jour des statistiques de l'époque
        epoch_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    # Calcul des métriques de l'époque
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    epoch_times.append(time.time() - epoch_start)

    # Évaluation sur le jeu de test
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    # affichage de la perte moyenne pour l'époque
    print(f"Epoch {epoch} | "
        f"Train loss: {avg_loss:.4f}, Train acc: {accuracy:.3f} | "
        f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.3f}")

#=================================================================================================================================================================
# Affichage des courbes de perte et de précision
#=================================================================================================================================================================
# temps total d'entraînement
total_training_time = time.time() - start_time
print(f"Temps total d'entraînement : {total_training_time:.2f} secondes")
# Évaluation sur le jeu de test
test_loss, test_acc = evaluate(model, test_loader, criterion)

test_losses.append(test_loss)
test_accuracies.append(test_acc)
# Courbe de pprécision
plt.figure()
plt.plot(train_accuracies, label="Train accuracy")
plt.plot(test_accuracies, label="Test accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy – agent supervisé")
plt.legend()
plt.grid()

# Courbe de loss
plt.figure()
plt.plot(train_losses, label="Train loss")
plt.plot(test_losses, label="Test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss – agent supervisé")
plt.legend()
plt.grid()


# Courbe du temps d'entraînement cumulé
plt.figure()
plt.plot(np.cumsum(epoch_times))
plt.xlabel("Epochs")
plt.ylabel("Temps cumulé (s)")
plt.title("Temps d'entraînement – agent supervisé")
plt.grid()
plt.show()

