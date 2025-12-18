#Dans ce fichier, nous implémentons l'algorithme Deep Q-Network (DQN) pour entraîner un agent à naviguer dans un environnement GridWorld.
#==================================================================================================================================
#Importations
#==================================================================================================================================
import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np 
import time
import matplotlib.pyplot as plt
from collections import deque
from GridWorld import GridWorld
from DQN import DQN
from ReplayBuffer import ReplayBuffer
import logging

logging.basicConfig(level=logging.INFO) 

#==================================================================================================================================
#Création des variables d'évaluations
#==================================================================================================================================
success_window = deque(maxlen=100)
success_rates = []
success_history = []
episode_rewards = []
losses = []
training_times = []
start_time = time.time()
#==================================================================================================================================
#Implémentation de la boucle d'entraînement DQN
#==================================================================================================================================
def train(): 
    # Initialisation de la seed pour reproductibilité (comme RL_course)
    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Initialisation de l'environnement, du réseau DQN, du buffer de relecture et des hyperparamètres
    env = GridWorld()
    # Détection du device optimal (MPS pour Mac, sinon CUDA, sinon CPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        logging.info("Utilisation du GPU MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Utilisation du GPU CUDA")
    else:
        device = torch.device('cpu')
        logging.info("Utilisation du CPU")
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    rb = ReplayBuffer()
    eps_start = 1.0
    eps_end = 0.05
    #eps_decay = 20000
    gamma = 0.95
    batch_size=64
    target_update=1000
    total_steps=0
    nb_episodes = 30000

    # Boucle principale d'entraînement 
    for episode in range(1, nb_episodes + 1):
        # Initialisation de l'épisode
        s=env.reset()
        done = False
        episode_reward = 0.0
        # Boucle de l'épisode
        while not done:
            total_steps += 1
            #eps = eps_end + (eps_start - eps_end) * np.exp(-1. * total_steps / eps_decay)
            #eps = max(eps_end, eps_start - (eps_start - eps_end) * (episode / nb_episodes))
            eps = max(eps_end, eps_start - (episode) / (nb_episodes))
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
            r = np.clip(r, -1.0, 1.0)
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
                # Calcul de la perte et optimisation du réseau
                criterion = nn.MSELoss()
                loss = criterion(q_values, target_q_values)
                losses.append(loss.item())
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            # Mise à jour du réseau cible
            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        # Fin de l'épisode : enregistrement des statistiques
        success = 1 if done and env.agent == env.goal else 0
        success_window.append(success)
        success_history.append(success)


        if len(success_window) == 100:
            success_rates.append(sum(success_window) / 100.0)
        else : 
            success_rates.append(sum(success_window) / len(success_window))

        episode_rewards.append(episode_reward)

        # Affichage des statistiques de l'épisode    
        if episode %10 == 0:            
            logging.info(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {eps:.3f}, Success Rate: {success_rates[-1]:.3f}, Loss: {loss.item() if 'loss' in locals() else 0.0:.4f}, Steps: {env.steps}")
        # Enregistrement du temps d'entraînement
        training_times.append(time.time() - start_time)
    # Sauvegarde du modèle entraîné
    torch.save(policy_net.state_dict(), 'dqn_model.pth')
    logging.info("\n✓ Modèle sauvegardé dans 'dqn_model.pth'")
    return device
# Lancement de l'entraînement
if __name__ == "__main__":
    device = train()
    end_time = time.time()
    training_time = end_time - start_time

    logging.info(f"Temps total d'entraînement : {training_time:.2f} secondes")
    logging.info(f"Device utilisé : {device}")
    global_success_rate = sum(success_history) / len(success_history)
    logging.info(f"Taux de succès global : {global_success_rate:.3f}")
    


#==================================================================================================================================
#Affichage des courbes d'évaluation
#==================================================================================================================================


# Sauvegarde des courbes pour affichage dans le notebook
import os
os.makedirs('results/03', exist_ok=True)
window = 100
avg_rewards = np.convolve(
    episode_rewards,
    np.ones(window)/window,
    mode="valid"
)
np.save('results/03/dqn_avg_rewards.npy', avg_rewards)
np.save('results/03/dqn_success_rates.npy', success_rates)
np.save('results/03/dqn_episode_rewards.npy', episode_rewards)
np.save('results/03/dqn_losses.npy', losses)
np.save('results/03/dqn_training_times.npy', training_times)

