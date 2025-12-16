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



# Fonction pour déterminer l'action optimale basée sur A* avec diversité
import random
def best_action(agent_pos, goal_pos, grid, size):
    """
    Trouve la première action d'un chemin optimal (A*) de agent_pos à goal_pos en évitant les obstacles.
    Si plusieurs chemins optimaux existent, choisit aléatoirement la première action parmi eux.
    Actions: 0=haut, 1=bas, 2=gauche, 3=droite
    """
    from collections import deque
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    actions = [0, 1, 2, 3]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # A* pour trouver tous les chemins optimaux
    queue = deque()
    queue.append((agent_pos, [], 0))  # (pos, path, cost)
    visited = {agent_pos: 0}
    min_cost = None
    optimal_first_actions = set()

    while queue:
        pos, path, cost = queue.popleft()
        if min_cost is not None and cost > min_cost:
            continue
        if pos == goal_pos:
            if min_cost is None:
                min_cost = cost
            if path:
                optimal_first_actions.add(path[0])
            continue
        for a in actions:
            nx, ny = pos[0] + dx[a], pos[1] + dy[a]
            if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] == 0:
                next_pos = (nx, ny)
                next_cost = cost + 1
                if next_pos not in visited or visited[next_pos] >= next_cost:
                    visited[next_pos] = next_cost
                    queue.append((next_pos, path + [a], next_cost))
    if optimal_first_actions:
        return random.choice(list(optimal_first_actions))
    # Si aucun chemin optimal trouvé, choisir une action valide au hasard
    valid_actions = []
    ax, ay = agent_pos
    for a in actions:
        nx, ny = ax + dx[a], ay + dy[a]
        if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] == 0:
            valid_actions.append(a)
    if valid_actions:
        return random.choice(valid_actions)
    return random.choice(actions)  # Si bloqué, action aléatoire


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
