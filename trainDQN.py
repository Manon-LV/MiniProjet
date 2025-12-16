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
def train(visualize=False, render_interval=100): 
    """
    Entraîne un agent DQN sur l'environnement GridWorld.
    
    Args:
        visualize (bool): Si True, affiche la visualisation pour certains épisodes.
        render_interval (int): Intervalle d'épisodes pour la visualisation (défaut: 100).
    """
    # Initialisation de l'environnement, du réseau DQN, du buffer de relecture et des hyperparamètres
    env= GridWorld()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = DQN().to(device)
    target_net = DQN() .to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    rb = ReplayBuffer()
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 20000
    gamma = 0.95
    batch_size=64
    target_update=1000
    total_steps=0

    # Boucle principale d'entraînement 
    for episode in range(1, 2001):
        # Initialisation de l'épisode
        s=env.reset()
        done = False
        episode_reward = 0.0
        
        # Visualisation pour certains épisodes
        should_render = visualize and (episode % render_interval == 0 or episode in [1, 10, 50])
        if should_render:
            print(f"\n🎬 Visualisation de l'épisode {episode}")
            env.render(episode=episode, reward=episode_reward, delay=0.1, epsilon=1.0)  # Epsilon initial
        
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
            r = np.clip(r, -1.0, 1.0)
            rb.push(s, a, r, s2, done)
            s = s2
            episode_reward += r
            
            # Visualisation si activée
            if should_render:
                env.render(episode=episode, reward=episode_reward, delay=0.05, epsilon=eps)
            
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
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {eps:.3f}")
        # Enregistrement du temps d'entraînement
        training_times.append(time.time() - start_time)
    
    # Fermeture de la visualisation si elle était active
    if visualize:
        env.close()
    
    # Sauvegarde du modèle entraîné
    torch.save(policy_net.state_dict(), 'dqn_model.pth')
    print("\n✓ Modèle sauvegardé dans 'dqn_model.pth'")
    
    return device
# Lancement de l'entraînement
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraînement DQN pour GridWorld')
    parser.add_argument('--visualize', action='store_true', 
                       help='Active la visualisation pendant l\'entraînement')
    parser.add_argument('--render-interval', type=int, default=100,
                       help='Intervalle d\'épisodes pour la visualisation (défaut: 100)')
    args = parser.parse_args()
    
    device = train(visualize=args.visualize, render_interval=args.render_interval)
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Temps total d'entraînement : {training_time:.2f} secondes")
    print(f"Device utilisé : {device}")
    global_success_rate = sum(success_history) / len(success_history)

    print(f"Taux de succès global : {global_success_rate:.3f}")
    


#==================================================================================================================================
#Affichage des courbes d'évaluation
#==================================================================================================================================

window = 100
avg_rewards = np.convolve(
    episode_rewards,
    np.ones(window)/window,
    mode="valid"
)

plt.figure()
plt.plot(avg_rewards)
plt.xlabel("Episodes")
plt.ylabel("Average reward (100 episodes)")
plt.title("Récompense moyenne glissante")
plt.grid()

# Courbe du taux de succès
plt.figure()
plt.plot(success_rates)
plt.xlabel("Episodes")
plt.ylabel("Success rate (moving average 100)")
plt.title("Taux de succès")
plt.grid()

# Courbe de la récompense par épisode
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Episode reward")
plt.title("Récompense par épisode")
plt.grid()

# Courbe de la perte DQN
plt.figure()
plt.plot(losses)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.title("Loss DQN")
plt.grid()

# Courbe du temps d'entraînement
plt.figure()
plt.plot(training_times)
plt.xlabel("Episodes")
plt.ylabel("Temps cumulé (s)")
plt.title("Temps d'entraînement")
plt.grid()
plt.show()

