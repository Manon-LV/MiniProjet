#==================================================================================================================================
#Class GridWorld
#==================================================================================================================================
#Cette classe implémente un environnement GridWorld simple pour les agents d'apprentissage par renforcement.
#L'environnement est une grille carrée où l'agent doit atteindre un objectif tout en évitant des obstacles.
#L'agent peut se déplacer dans quatre directions (haut, bas, gauche, droite).
#L'état de l'environnement est représenté par une matrice 2D où chaque cellule 
#peut être vide, contenir un obstacle ou représenter la position de l'agent ou de l'objectif.
#L'agent reçoit des récompenses en fonction de ses actions : une récompense positive pour atteindre l'objectif,
#une pénalité pour heurter un obstacle, et une petite pénalité pour chaque étape prise.


# L'environnement est de type Markovien (MDP) : l'observation courante
# contient toute l'information nécessaire pour la prise de décision.



#==================================================================================================================================
#Importations
#==================================================================================================================================
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#==================================================================================================================================
#Implémentation de la classe GridWorld
#==================================================================================================================================
class GridWorld:



    #Constructeur de la classe GridWorld
    def __init__(self, size=10, n_obstacle=5, max_step=200, repeat_penalty: float = -0.5):
        ''' 
        Constructeur de la classe GridWorld.
        inputs:
            size (int): Taille de la grille (la grille a une forme carrée) (size x size).
            n_obstacle (int): Nombre d'obstacles dans la grille.
            max_step (int): Nombre maximum d'étapes par épisode.
            repeat_penalty (float): Pénalité appliquée lorsqu'une case déjà visitée est revisitée durant l'épisode.
        '''
        self.size = size
        self.n_obstacle = n_obstacle
        self.max_step = max_step
        self.repeat_penalty = repeat_penalty
        self.fig = None
        self.ax = None
        self.reset() 



    #fonction de réinitialisation de l'environnement
    def reset(self):
        ''' 
        Réinitialise l'environnement GridWorld. C'est à dire, place l'agentà un point aléatoire, 
        génère des obstacles aléatoires et place l'objectif à une position aléatoire. 0 signifie une cellule vide,
        1 signifie un obstacle.
        outputs:
            obs (np.array): Observation initiale de l'environnement après réinitialisation.
            C'est à dire une matrice 3D où chaque couche représente l'agent, les obstacles et l'objectif.
        '''
        # initialisation de la grille : vide
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.grid[:]=0
        buffer_obs = 0
        #Ajout des obstacles dans la grille de manière aléatoire
        while buffer_obs < self.n_obstacle:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            if self.grid[y][x] == 0 :
                self.grid[y][x] = 1 
                buffer_obs += 1
        # Placement de l'agent à une position aléatoire
        while True:
            ax,ay = np.random.randint(0,self.size,2)
            if self.grid[ax,ay]==0:
                break
        self.agent = (ax,ay)
        # Placement de l'objectif à une position aléatoire
        while True: 
            gx, gy = np.random.randint(0, self.size, size=2)
            if self.grid[gy][gx] == 0 and (gx,gy)!=self.agent:
                break
        self.goal = (gx, gy)
        self.steps = 0
        # Suivi des cases visitées pendant l'épisode
        self.visited = set()
        self.visited.add(self.agent)
        return self.__get_obs()



    #fonction privée pour obtenir l'observation actuelle
    def __get_obs(self):
        '''
        Génère l'observation actuelle de l'environnement.
        L'état est encodé sous forme de tenseur 3 x size x size :
         - canal 0 : position de l'agent
         - canal 1 : obstacles
         - canal 2 : objectif
        outputs:
            obs (np.array): Observation actuelle de l'environnement.
            C'est à dire une matrice 3D où chaque couche représente l'agent, les obstacles et l'objectif.
        '''
        # initialisation de l'observation
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        # position de l'agent
        ax, ay = self.agent
        # position de l'objectif
        gx, gy = self.goal
        # mise à jour de l'observation
        obs[0, ay, ax] = 1.0
        obs[1, self.grid==1] = 1.0
        obs[2, gy, gx] = 1.0
        return obs
    


    #fonction pour effectuer une action dans l'environnement
    def step(self, action):
        '''
        Effectue une action dans l'environnement GridWorld. C'est à dire, déplace l'agent dans la direction spécifiée par l'action.
        En cas de collision avec un obstacle, l'épisode est terminé afin de pénaliser fortement les trajectoires invalides.
        inputs:
            action (int): Action à effectuer (0: haut, 1: bas, 2: gauche, 3: droite).
        outputs:
            obs (np.array): Observation après l'action.
            reward (float): Récompense obtenue après l'action. 0.1 pour chaque étape, -5 pour heurter un obstacle, 10 pour atteindre l'objectif.
            done (bool): Indique si l'épisode est terminé.
            info (dict): Informations supplémentaires (vide dans ce cas).
        '''
        # Déplacement de l'agent en fonction de l'action
        dx = [-1,1,0,0]
        dy = [0,0,-1,1]
        # position actuelle de l'agent
        ax, ay = self.agent
        # nouvelle position de l'agent
        nx, ny = ax + dx[action], ay + dy[action]
        #Ajout du compteur de pas
        self.steps +=1
        # Vérification des limites de la grille
        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
            nx, ny = ax, ay
        # Vérification des obstacles
        if self.grid[ny][nx] == 1:
            reward = -5.0
            done = True
            self.agent = (nx, ny)
            return self.__get_obs(), reward, done, {}
        # Pénalité si la case a déjà été visitée pendant l'épisode
        revisit_penalty = self.repeat_penalty if (nx, ny) in getattr(self, 'visited', set()) else 0.0

        self.agent = (nx, ny)
        # Enregistrer la visite de la case actuelle
        if hasattr(self, 'visited'):
            self.visited.add(self.agent)
        # Vérification de l'objectif
        if self.agent == self.goal:
            return self.__get_obs(), 100.0, True, {}
        # Vérification du nombre maximum de pas
        if self.steps >= self.max_step:
            return self.__get_obs(), -1, True, {}
        # Shaping : pénalité distance à l'objectif (distance de Manhattan)
        gx, gy = self.goal
        dist = abs(gx - nx) + abs(gy - ny)
        distance_penalty = -0.05 * dist
        # Récompense avec shaping + pénalité de revisite
        reward = -0.1 + distance_penalty + revisit_penalty
        return self.__get_obs(), reward, False, {}


    #fonction pour visualiser l'environnement
    def render(self, episode=None, reward=None, delay=0.1, epsilon=None):
        '''
        Visualise l'environnement GridWorld avec matplotlib.
        inputs:
            episode (int): Numéro de l'épisode actuel (optionnel).
            reward (float): Récompense cumulée de l'épisode (optionnel).
            delay (float): Délai entre les frames en secondes (0.1 par défaut).
            epsilon (float): Valeur d'epsilon-greedy (optionnel).
        '''
        if self.fig is None:
            plt.ion()  # Mode interactif
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax.set_xticks(np.arange(0, self.size, 1))
        self.ax.set_yticks(np.arange(0, self.size, 1))
        
        # Dessiner les obstacles (gris foncé)
        for y in range(self.size):
            for x in range(self.size):
                if self.grid[y][x] == 1:
                    rect = Rectangle((x, self.size - 1 - y), 1, 1, 
                                   facecolor='#2c3e50', edgecolor='black', linewidth=1)
                    self.ax.add_patch(rect)
        
        # Dessiner l'objectif (vert)
        gx, gy = self.goal
        goal_rect = Rectangle((gx, self.size - 1 - gy), 1, 1, 
                             facecolor='#2ecc71', edgecolor='black', linewidth=2)
        self.ax.add_patch(goal_rect)
        self.ax.text(gx + 0.5, self.size - 1 - gy + 0.5, '★', 
                    fontsize=30, ha='center', va='center', color='white')
        
        # Dessiner l'agent (bleu)
        ax, ay = self.agent
        agent_rect = Rectangle((ax, self.size - 1 - ay), 1, 1, 
                              facecolor='#3498db', edgecolor='black', linewidth=2)
        self.ax.add_patch(agent_rect)
        self.ax.text(ax + 0.5, self.size - 1 - ay + 0.5, '●', 
                    fontsize=30, ha='center', va='center', color='white')
        
        # Afficher les métriques
        title = f'GridWorld {self.size}x{self.size}'
        if episode is not None:
            title += f' | Episode: {episode}'
        title += f' | Steps: {self.steps}/{self.max_step}'
        if reward is not None:
            title += f' | Reward: {reward:.2f}'
        if epsilon is not None:
            title += f' | Eps: {epsilon:.3f}'
        
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.pause(delay)
        plt.draw()
    


    def close(self):
        '''
        Ferme la fenêtre de visualisation.
        '''
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
