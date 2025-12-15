#=-===============================================================================================================================================================================
# Génération d'un dataset pour un agent dans un environnement GridWorld
#=================================================================================================================================================================================
# Ce script génère un dataset d'observations et d'actions optimales pour un agent dans un environnement GridWorld.
# L'agent est placé dans une grille avec des obstacles et un objectif, et l'action optimale est déterminée en fonction de la position relative de 
# l'agent et de l'objectif.
# Le dataset est sauvegardé sous forme de fichiers numpy pour une utilisation ultérieure dans l'entraînement de modèles d'apprentissage automatique.


#=================================================================================================================================================================================
# Importations
#=================================================================================================================================================================================
import numpy as np
from GridWorld import GridWorld



# Fonction pour déterminer l'action optimale basée sur la position de l'agent et de l'objectif
def best_action(agent_pos, goal_pos):
    ax, ay = agent_pos
    gx, gy = goal_pos
    if abs(gx - ax) > abs(gy - ay):
        return 3 if gx > ax else 2
    else:
        return 1 if gy > ay else 0


#=================================================================================================================================================================================
# Génération du dataset
#=================================================================================================================================================================================
# Initialisation de l'environnement GridWorld
env = GridWorld()
# Listes pour stocker les observations et les actions
X, y = [], []

# Boucle pour générer des échantillons de données
for _ in range(20000):
    obs = env.reset()
    action = best_action(env.agent, env.goal)
    X.append(obs)
    y.append(action)

# Conversion des listes en tableaux numpy 
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Sauvegarde des datasets sous forme de fichiers numpy
np.save("X.npy", X)
np.save("y.npy", y)
print("Dataset généré :", X.shape, y.shape)
