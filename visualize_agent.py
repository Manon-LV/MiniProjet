#!/usr/bin/env python3
"""
Script de visualisation pour observer un agent DQN entraîné en action.
Permet de voir comment l'agent se déplace dans la grille en temps réel.
"""

#==================================================================================================================================
#Importations
#==================================================================================================================================
import torch
import numpy as np
import argparse
from GridWorld import GridWorld
from DQN import DQN
import logging

logging.basicConfig(level=logging.INFO) 

#==================================================================================================================================
#Fonction principale de visualisation
#==================================================================================================================================
def visualize_agent(model_path='dqn_model.pth', n_episodes=100, delay=1, size=10, n_obstacle=5):
    """
    Visualise un agent DQN entraîné en action.
    
    Args:
        model_path (str): Chemin vers le modèle sauvegardé.
        n_episodes (int): Nombre d'épisodes à visualiser.
        delay (float): Délai entre chaque frame (en secondes).
        size (int): Taille de la grille.
        n_obstacle (int): Nombre d'obstacles.
    """
    # Initialisation de l'environnement et du modèle
    env = GridWorld(size=size, n_obstacle=n_obstacle)
    # Détection du device optimal (MPS pour Mac, sinon CUDA, sinon CPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Chargement du modèle
    policy_net = DQN().to(device)
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.eval()
        logging.info(f"✓ Modèle chargé depuis {model_path}")
    except FileNotFoundError:
        logging.warning(f"⚠ Modèle non trouvé à {model_path}. Utilisation d'un agent aléatoire.")
        policy_net = None
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Visualisation de l'agent DQN")
    logging.info(f"{'='*60}")
    logging.info(f"Nombre d'épisodes: {n_episodes}")
    logging.info(f"Grille: {size}x{size} avec {n_obstacle} obstacles")
    logging.info(f"Device: {device}")
    logging.info(f"{'='*60}\n")
    
    # Statistiques globales
    total_successes = 0
    total_rewards = []
    total_steps = []
    
    try:
        for episode in range(1, n_episodes + 1):
            state = env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            
            logging.info(f"\n🎮 Episode {episode}/{n_episodes}")
            env.render(episode=episode, reward=episode_reward, delay=delay)
            
            while not done:
                # Sélection de l'action
                if policy_net is not None:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        q_values = policy_net(state_tensor)
                        action = int(q_values.argmax(1).item())
                else:
                    # Agent aléatoire si pas de modèle
                    action = np.random.randint(0, 4)
                
                # Exécution de l'action
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Visualisation
                env.render(episode=episode, reward=episode_reward, delay=delay)
                
                state = next_state
            
            # Statistiques de l'épisode
            success = env.agent == env.goal
            total_successes += success
            total_rewards.append(episode_reward)
            total_steps.append(step_count)
            
            status = "✓ SUCCÈS" if success else "✗ ÉCHEC"
            logging.info(f"  {status} | Reward: {episode_reward:.2f} | Steps: {step_count}")
        
        # Affichage des statistiques finales
        logging.info(f"\n{'='*60}")
        logging.info(f"📊 Statistiques globales")
        logging.info    (f"{'='*60}")
        logging.info(f"Taux de succès: {total_successes}/{n_episodes} ({100*total_successes/n_episodes:.1f}%)")
        logging.info(f"Reward moyen: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        logging.info(f"Steps moyen: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
        logging.info(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        logging.info("\n\n⚠ Visualisation interrompue par l'utilisateur.")
    finally:
        env.close()
        logging.info("Fenêtre de visualisation fermée.")


#==================================================================================================================================
#Point d'entrée du script
#==================================================================================================================================
if __name__ == "__main__":
    pathToModel = 'dqn_model.pth'
    parser = argparse.ArgumentParser(description="Visualise un agent DQN en action dans GridWorld")
    parser.add_argument('--model', type=str, default= pathToModel, 
                       help=f'Chemin vers le modèle sauvegardé (défaut: {pathToModel})')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Nombre d\'épisodes à visualiser (défaut: 100)')
    parser.add_argument('--delay', type=float, default=1, 
                       help='Délai entre les frames en secondes (défaut: 1)')
    parser.add_argument('--size', type=int, default=10, 
                       help='Taille de la grille (défaut: 10)')
    parser.add_argument('--obstacles', type=int, default=5, 
                       help='Nombre d\'obstacles (défaut: 5)')
    
    args = parser.parse_args()
    
    visualize_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        delay=args.delay,
        size=args.size,
        n_obstacle=args.obstacles
    )
