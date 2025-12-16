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


# Fonction pour vérifier si une action est valide (ne mène pas à une collision avec un obstacle ou hors de la grille)
def is_valid_action(env, action):
    # Récupération de la position actuelle de l'agent
    ax, ay = env.agent
    # Définition des déplacements possibles pour chaque action
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    # Calcul de la nouvelle position de l'agent après l'action
    nx, ny = ax + dx[action], ay + dy[action]
    # si hors grille
    if nx < 0 or nx >= env.size or ny < 0 or ny >= env.size:
        return False
    # si obstacle
    if env.grid[ny][nx] == 1:
        return False
    return True


# Fonction pour déterminer l'action optimale basée sur la position de l'agent et de l'objectif
def best_action_with_obstacles(env):
    # Récupération des positions de l'agent et de l'objectif
    ax, ay = env.agent
    gx, gy = env.goal
    # Liste des actions valides (à gauche, à droite, en haut, en bas)
    valid_actions = [a for a in range(4) if is_valid_action(env, a)]

    # si aucune action n'est valide (l'agent est bloqué par les obstacles ou les murs)
    if len(valid_actions) == 0:
        return None

    # Choix de l'action qui minimise la distance de Manhattan à l'objectif
    best_score = -1e9
    best_actions = []
    # Évaluation de chaque action valide
    for a in valid_actions:
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        # Calcul de la nouvelle position après l'action
        nx, ny = ax + dx[a], ay + dy[a]

        # Calcul de la distance de Manhattan avant et après l'action
        old_dist = abs(gx - ax) + abs(gy - ay)
        new_dist = abs(gx - nx) + abs(gy - ny)
        score = old_dist - new_dist
        
        # Sélection de l'action avec le meilleur score
        if score > best_score:
            # Mise à jour du meilleur score et des actions associées
            best_score = score
            best_actions = [a]
        # Si plusieurs actions ont le même score, les ajouter à la liste
        elif score == best_score:
            best_actions.append(a)
    # Retourner les meilleures actions
    return best_actions

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
    actions = best_action_with_obstacles(env)
    y_dist = np.zeros(4, dtype=np.float32)
    if actions is not None:
        for a in actions:
            y_dist[a] = 1.0 / len(actions)
        X.append(obs)
        y.append(y_dist)

# Conversion des listes en tableaux numpy 
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
y_dist = y_dist + 1e-8
y_dist /= y_dist.sum()
# Sauvegarde des datasets sous forme de fichiers numpy
np.save("X.npy", X)
np.save("y.npy", y)
print("Dataset généré :", X.shape, y.shape)
