#Dans ce fichier, nous implémentons l'algorithme Deep Q-Network (DQN) pour entraîner un agent à naviguer dans un environnement GridWorld.
#==================================================================================================================================
#Importations
#==================================================================================================================================
import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from GridWorld import GridWorld
from DQN import DQN
from ReplayBuffer import ReplayBuffer
#==================================================================================================================================
#Implémentation de la boucle d'entraînement DQN
#==================================================================================================================================
def train(): 
    # Initialisation de l'environnement, du réseau DQN, du buffer de relecture et des hyperparamètres
    env= GridWorld()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = DQN().to(device)
    target_net = DQN() .to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    rb = ReplayBuffer()
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 20000
    gamma = 0.99
    batch_size=64
    target_update=1000
    total_steps=0

    # Boucle principale d'entraînement 
    for episode in range(1, 2001):
        # Initialisation de l'épisode
        s=env.reset()
        done = False
        episode_reward = 0.0
        # Boucle de l'épisode
        while not done:
            total_steps += 1
            eps = eps_end + (eps_start - eps_end) * np.exp(-1. * total_steps / eps_decay)
            # Sélection de l'action selon une politique epsilon-greedy
            if random.random() < eps:
                a = random.randrange(4)
            else:
                with torch.no_grad():
                   st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                   q_values = policy_net(st)
                   a = int(q_values.argmax(1).item())
            # Exécution de l'action dans l'environnement
            s2, r, done, _ = env.step(a)
            rb.push(s, a, r, s2, done)
            s = s2
            episode_reward += r
            # Mise à jour du réseau DQN
            if len(rb) >= batch_size:
                batch = rb.sample(batch_size)
                s_batch = torch.tensor(np.array(batch.s), dtype=torch.float32).to(device)
                a_batch = torch.tensor(batch.a, dtype=torch.long).unsqueeze(1).to(device)
                r_batch = torch.tensor(batch.r, dtype=torch.float32).unsqueeze(1).to(device)
                s2_batch = torch.tensor(np.array(batch.s2), dtype=torch.float32).to(device)
                done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)
                
                # Calcul des valeurs Q et de la perte
                q_values = policy_net(s_batch).gather(1, a_batch)
                with torch.no_grad():
                    next_q_values = target_net(s2_batch).max(1)[0].unsqueeze(1)
                    target_q_values = r_batch + gamma * next_q_values * (1.0 - done_batch)
                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            # Mise à jour du réseau cible
            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Affichage des statistiques de l'épisode    
        if episode %10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {eps:.3f}")


# Lancement de l'entraînement
if __name__ == "__main__":
    train()