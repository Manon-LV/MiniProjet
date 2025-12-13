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

#==================================================================================================================================
#Implémentation de la classe GridWorld
#==================================================================================================================================
class GridWorld:



    #Constructeur de la classe GridWorld
    def __init__(self, size=10, n_obstacle=15, max_step=200):
        ''' 
        Constructeur de la classe GridWorld.
        inputs:
            size (int): Taille de la grille (la grille a une forme carrée) (size x size).
            n_obstacle (int): Nombre d'obstacles dans la grille.
            max_step (int): Nombre maximum d'étapes par épisode.
        '''
        self.size = size
        self.n_obstacle = n_obstacle
        self.max_step = max_step
        self.reset() 



    #fonction de réinitialisation de l'environnement
    def reset(self):
        ''' 
        Réinitialise l'environnement GridWorld. C'est à dire, place l'agent au point de départ (0,0), 
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
            if self.grid[y][x] == 0 and (x,y)!=(0,0):
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
        self.agent = (nx, ny)
        # Vérification de l'objectif
        if self.agent == self.goal:
            return self.__get_obs(), 10.0, True, {}
        # Vérification du nombre maximum de pas
        if self.steps >= self.max_step:
            return self.__get_obs(), -1.0, True, {}
        return self.__get_obs(), -0.1, False, {}