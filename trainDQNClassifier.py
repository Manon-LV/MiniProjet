import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ReplayBuffer import ReplayBuffer
from DQNClassifier import DQNClassifier
from ClassificationEnv import ClassificationEnv
import torch.utils.data
import logging

logging.basicConfig(level=logging.INFO)

def train_dqn_classifier():
    # Chargement des données

    X = np.load('X_20000.npy')
    y = np.load('y_20000.npy')
    X = X.astype(np.float32)
    # Si X est 3D (ex: images), on aplatit chaque exemple
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    # Split train/test
        # Création du TensorDataset
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        # Extraction numpy pour l'env RL
        X_train = X_tensor[train_dataset.indices].numpy()
        y_train = y_tensor[train_dataset.indices].numpy()
        X_test = X_tensor[test_dataset.indices].numpy()
        y_test = y_tensor[test_dataset.indices].numpy()
    env = ClassificationEnv(X_train, y_train)
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y))

    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device utilisé : {device}")

    # Réseaux
    policy_net = DQNClassifier(input_dim, n_classes).to(device)
    target_net = DQNClassifier(input_dim, n_classes).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    rb = ReplayBuffer()
    gamma = 0.99
    batch_size = 64
    target_update = 100
    nb_episodes = 10
    total_steps = 0
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 5000
    losses = []
    accuracy_history = []
    test_accuracy_history = []

    for episode in range(nb_episodes):
        obs = env.reset()
        done = False
        correct = 0
        total = 0
        while not done:
            total_steps += 1
            eps = eps_end + (eps_start - eps_end) * np.exp(-1. * total_steps / eps_decay)
            if np.random.rand() < eps:
                action = np.random.randint(n_classes)
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(obs_tensor)
                    action = int(q_values.argmax(1).item())
            next_obs, reward, done, info = env.step(action)
            rb.push(obs, action, reward, next_obs if next_obs is not None else np.zeros_like(obs), done)
            obs = next_obs
            if info['label'] == info['pred']:
                correct += 1
            total += 1
            # Apprentissage
            if len(rb) >= batch_size:
                batch = rb.sample(batch_size)
                s_batch = torch.tensor(np.array(batch.s), dtype=torch.float32).to(device)
                a_batch = torch.tensor(batch.a, dtype=torch.long).unsqueeze(1).to(device)
                r_batch = torch.tensor(batch.r, dtype=torch.float32).unsqueeze(1).to(device)
                s2_batch = torch.tensor(np.array(batch.s2), dtype=torch.float32).to(device)
                done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)
                q_values = policy_net(s_batch).gather(1, a_batch)
                with torch.no_grad():
                    next_q_values = target_net(s2_batch).max(1)[0].unsqueeze(1)
                    target_q_values = r_batch + gamma * next_q_values * (1.0 - done_batch)
                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
        acc = correct / total
        accuracy_history.append(acc)
        # Évaluation sur le test set
        policy_net.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            q_test = policy_net(X_test_tensor)
            y_pred = q_test.argmax(1).cpu().numpy()
            test_acc = (y_pred == y_test).mean()
            test_accuracy_history.append(test_acc)
        policy_net.train()
        logging.info(f"Episode {episode+1}/{nb_episodes} | Train acc: {acc:.4f} | Test acc: {test_acc:.4f}")
    torch.save(policy_net.state_dict(), 'dqn_classifier_model.pth')
    logging.info("Modèle DQNClassifier sauvegardé dans 'dqn_classifier_model.pth'")
    return accuracy_history, losses

if __name__ == "__main__":
    train_dqn_classifier()
