import torch
import numpy as np
import matplotlib.pyplot as plt
from GridWorld import GridWorld
from DQN import DQN

def test_dqn(model_path, n_episodes=1000, size=10, n_obstacle=5):
    device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = DQN().to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    env = GridWorld(size=size, n_obstacle=n_obstacle)
    episode_rewards, success_rates, steps_per_episode = [], [], []
    success_window = []

    for episode in range(n_episodes):
        s = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            with torch.no_grad():
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = policy_net(st)
                a = int(q_values.argmax(1).item())
            s, r, done, _ = env.step(a)
            total_reward += r
            steps += 1
        episode_rewards.append(total_reward)
        steps_per_episode.append(steps)
        success = 1 if env.agent == env.goal else 0
        success_window.append(success)
        if len(success_window) > 100:
            success_window.pop(0)
        success_rates.append(np.mean(success_window))

    # Sauvegarde des courbes pour affichage dans le notebook
    import os
    os.makedirs('results/03', exist_ok=True)
    window = 100
    avg_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    np.save('results/03/testdqn_avg_rewards.npy', avg_rewards)
    np.save('results/03/testdqn_success_rates.npy', np.array(success_rates))
    np.save('results/03/testdqn_episode_rewards.npy', np.array(episode_rewards))
    np.save('results/03/testdqn_steps_per_episode.npy', np.array(steps_per_episode))

if __name__ == '__main__':
    test_dqn('dqn_model03.pth', n_episodes=1000)