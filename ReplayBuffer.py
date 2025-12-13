#==================================================================================================================================
# Classe ReplayBuffer
#==================================================================================================================================
# Cette classe implémente un buffer de relecture (Replay Buffer) pour stocker les transitions d'expérience
# dans un environnement d'apprentissage par renforcement. Le Replay Buffer permet de stocker des transitions
# sous la forme de tuples (état, action, récompense, état suivant, done) et de les échantillonner aléatoirement
# pour entraîner un agent. Cela aide à briser la corrélation temporelle entre les expériences et améliore la stabilité
# de l'apprentissage.
#==================================================================================================================================
# Importations
#==================================================================================================================================
from collections import deque, namedtuple
import random

#==================================================================================================================================
# Implémentation de la classe ReplayBuffer
#==================================================================================================================================

# Définition de la structure de données pour une transition : 
#s : état actuel, a : action prise, r : récompense reçue, s2 : état suivant, done : indicateur de fin d'épisode
Transition = namedtuple('Transition', ('s','a','r','s2','done'))


class ReplayBuffer:
    # Constructeur de la classe ReplayBuffer
    def __init__(self, capacity=100000):
        '''
        Constructeur de la classe ReplayBuffer.
        inputs:
            capacity (int): Capacité maximale du buffer.
        '''
        # Initialisation du buffer comme une deque(file d'attente à double extrémité) avec une taille maximale
        self.buffer = deque(maxlen=capacity)

    # Méthode pour ajouter une transition au buffer
    def push(self, *args):
        '''
        Ajoute une transition au buffer: c'est à dire, un tuple (s, a, r, s2, done).
        inputs:
            *args: Arguments de la transition (s, a, r, s2, done).
        '''
        # Ajout de la transition au buffer
        self.buffer.append(Transition(*args))


    # Méthode pour échantillonner un batch de transitions aléatoires du buffer
    def sample(self, batch_size):
        '''
        Échantillonne un batch de transitions aléatoires du buffer.
        inputs:
            batch_size (int): Taille du batch à échantillonner.
        outputs:
            Transition: Un batch de transitions échantillonnées.
        '''
        # Échantillonnage aléatoire d'un batch de transitions
        batch=random.sample(self.buffer, batch_size)
        # Retourne le batch sous forme de Transition
        return Transition(*zip(*batch))

    # Méthode pour obtenir la taille actuelle du buffer
    def __len__(self):
        '''
        Retourne la taille actuelle du buffer.
        outputs:
            int: Taille actuelle du buffer.
        '''
        return len(self.buffer)
  