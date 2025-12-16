#=================================================================================================================================================================
#CNN.py
#=================================================================================================================================================================
# Ce fichier définit une architecture de réseau de neurones convolutifs (CNN) pour une politique d'agent dans un environnement GridWorld.
# Le réseau prend en entrée une observation de l'environnement (par exemple, une image représentant l'état actuel de la grille)
# et produit des probabilités pour chaque action possible.  
#=================================================================================================================================================================================
#Importations
#=================================================================================================================================================================================
import torch.nn as nn
import torch.nn.functional as F
#=================================================================================================================================================================================
#CNN
#=================================================================================================================================================================================
# class CNN(nn.Module):
#     #Constructeur de la classe CNN
#     def __init__(self):
#         # Appel du constructeur de la classe parente nn.Module
#         super().__init__()
#         # Définition des couches du réseau
#         # Couches convolutionnelles pour extraire les caractéristiques de l'entrée
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         # Deuxième couche convolutionnelle pour extraire des caractéristiques plus complexes
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         # Dropout après les convolutions
#         self.dropout1 = nn.Dropout2d(0.2)
#         self.dropout2 = nn.Dropout2d(0.3)
#         # Couches entièrement connectées pour produire les probabilités d'action
#         self.fc1 = nn.Linear(32 * 10 * 10, 128)
#         self.dropout_fc = nn.Dropout(0.5)
#         # Couche de sortie pour produire les scores pour chaque action
#         self.fc2 = nn.Linear(128, 4)

#     # Fonction de passage en avant du réseau
#     def forward(self, x):
#         # Passe en avant du réseau CNN.
#         # inputs:   
#         #     x (torch.Tensor): Entrée du réseau (observation de l'environnement).
#         # outputs:
#         #     torch.Tensor: Sortie du réseau (scores pour chaque action).
#         # Passage à travers les couches convolutionnelles avec activation ReLU et Dropout
#         x = F.relu(self.conv1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.conv2(x))
#         x = self.dropout2(x)
#         # Aplatissement des sorties des couches convolutionnelles
#         x = x.view(x.size(0), -1)
#         # Passage à travers les couches entièrement connectées avec activation ReLU et Dropout
#         x = F.relu(self.fc1(x))
#         x = self.dropout_fc(x)
#         return self.fc2(x)

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)