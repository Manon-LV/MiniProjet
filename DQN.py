#==================================================================================================================================
#Classe DQN pour l'approximation de la fonction Q dans un environnement GridWorld
#==================================================================================================================================
# Cette classe implémente un réseau de neurones convolutif profond (DQN) pour approximer la fonction Q dans un environnement GridWorld.
# Le réseau prend en entrée une observation de l'environnement (par exemple, une image représentant l'état actuel de la grille) 
# et produit des valeurs Q pour chaque action possible.
# Le réseau est composé de couches convolutionnelles suivies de couches entièrement connectées.
# La fonction Q est utilisée pour guider l'agent dans la prise de décision afin de maximiser la récompense cumulative dans l'environnement.
# En effet,  la fonction Q (fonction d'action-valeur) correspond à l'esperance de la récompense cumulée future associée à une action dans
# un état donné.
# La politique de l'agent est dérivée de ces valeurs Q en sélectionnant l'action maximisant Q(s,a) (argmax),
# éventuellement avec exploration epsilon-greedy.
#==================================================================================================================================
#Importations
#==================================================================================================================================
import random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

#==================================================================================================================================
#Implémentation de la classe DQN
#==================================================================================================================================
class DQN(nn.Module):
    def __init__(self, in_channel=3, n_actions = 4) :
        '''
        Constructeur de la classe DQN.
        inputs:
            in_channel (int): Nombre de canaux en entrée
            n_actions (int): Nombre d'actions possibles.
        '''
        # Appel du constructeur de la classe parente nn.Module
        super().__init__()
        # Définition des couches du réseau
        self.conv=nn.Sequential(
            #couches convolutionnelles pour extraire les caractéristiques de l'entrée
            nn.Conv2d(in_channel, 16, kernel_size=3, padding=1),
            #couche d'activation ReLU pour introduire la non-linéarité
            nn.ReLU(),
            # deuxième couche convolutionnelle pour extraire des caractéristiques plus complexes
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # couche d'activation ReLU pour introduire la non-linéarité
            nn.ReLU())
        # Couches entièrement connectées pour produire les valeurs Q pour chaque action
        self.fc=nn.Sequential(
            #couche de flattening pour aplatir les sorties des couches convolutionnelles
            nn.Flatten(),
            # première couche entièrement connectée pour combiner les caractéristiques extraites
            nn.Linear(32*10*10, 128),
            # couche d'activation ReLU pour introduire la non-linéarité
            nn.ReLU(),
            # couche de sortie pour produire les valeurs Q pour chaque action
            nn.Linear(128, n_actions)
        )



     # Fonction de passage en avant du réseau
    def forward(self, x):
        '''
        Passe en avant du réseau DQN.
        inputs:
            x (torch.Tensor): Entrée du réseau (par exemple, une image).
        outputs:
            torch.Tensor: Valeurs Q pour chaque action.
        '''
        # Passage à travers les couches convolutionnelles puis les couches entièrement connectées
        return self.fc(self.conv(x))