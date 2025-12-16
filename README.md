# MiniProjet - Deep Q-Network (DQN) pour GridWorld

Implémentation d'un agent DQN qui apprend à naviguer dans un environnement GridWorld pour atteindre un objectif tout en évitant des obstacles.

## 📋 Description

L'agent utilise l'algorithme Deep Q-Network (DQN) pour apprendre une politique optimale de navigation dans une grille. L'environnement comprend :
- Un agent qui doit atteindre une cible
- Des obstacles à éviter
- Un système de récompenses pour guider l'apprentissage

## 🚀 Installation

```bash
pip install torch numpy matplotlib
```

## 📖 Utilisation

### 1. Entraînement de l'agent

**Entraînement standard (sans visualisation) :**
```bash
python trainDQN.py
```

**Entraînement avec visualisation :**
```bash
python trainDQN.py --visualize --render-interval 100
```

Options disponibles :
- `--visualize` : Active la visualisation en temps réel pendant l'entraînement
- `--render-interval N` : Visualise tous les N épisodes (défaut: 100)

**Exemples :**
```bash
# Visualiser tous les 50 épisodes
python trainDQN.py --visualize --render-interval 50

# Visualiser fréquemment (tous les 10 épisodes)
python trainDQN.py --visualize --render-interval 10
```

⚠️ **Note :** La visualisation ralentit l'entraînement. Utilisez des intervalles larges (100-500) pour l'entraînement complet.

### 2. Visualisation d'un agent entraîné

**Visualisation basique :**
```bash
python visualize_agent.py
```

**Avec options personnalisées :**
```bash
python visualize_agent.py --episodes 10 --delay 0.2
```

Options disponibles :
- `--model PATH` : Chemin vers le modèle entraîné (défaut: `dqn_model.pth`)
- `--episodes N` : Nombre d'épisodes à visualiser (défaut: 5)
- `--delay SEC` : Délai entre les frames en secondes (défaut: 0.3)
- `--size N` : Taille de la grille (défaut: 10)
- `--obstacles N` : Nombre d'obstacles (défaut: 5)

**Exemples :**
```bash
# Visualiser 20 épisodes lentement pour analyse détaillée
python visualize_agent.py --episodes 20 --delay 0.5

# Visualisation rapide de 5 épisodes
python visualize_agent.py --episodes 5 --delay 0.1

# Tester sur une grille plus grande
python visualize_agent.py --size 15 --obstacles 10 --episodes 10

# Utiliser un modèle spécifique
python visualize_agent.py --model mon_modele.pth --episodes 10
```

### 3. Affichage des résultats d'entraînement

```bash
python plot_results.py
```

Affiche les courbes de performance de l'entraînement.

## 📊 Visualisation

La visualisation en temps réel affiche :
- 🔵 **Agent** : Position actuelle (cercle bleu)
- ⭐ **Objectif** : Cible à atteindre (étoile verte)
- ⬛ **Obstacles** : Obstacles dans la grille (carrés gris foncé)
- 📈 **Métriques** : Numéro d'épisode, steps effectués, reward cumulé

### Interprétation des récompenses
- `+10.0` : Atteindre l'objectif
- `-5.0` : Heurter un obstacle
- `-0.01` : Chaque step (encourage l'efficacité)
- `-0.5` : Timeout (limite de 200 steps atteinte)

## 🎯 Workflow recommandé

```bash
# 1. Entraîner l'agent (entraînement complet sans visualisation)
python trainDQN.py

# 2. Visualiser les performances avec les graphiques
python plot_results.py

# 3. Observer l'agent en action
python visualize_agent.py --episodes 10 --delay 0.2

# 4. Déboguer avec visualisation pendant l'entraînement (optionnel)
python trainDQN.py --visualize --render-interval 200
```

## 📁 Structure du projet

```
.
├── GridWorld.py          # Environnement de simulation
├── DQN.py               # Architecture du réseau de neurones
├── ReplayBuffer.py      # Buffer d'expérience pour DQN
├── trainDQN.py          # Script d'entraînement principal
├── visualize_agent.py   # Script de visualisation
├── plot_results.py      # Affichage des résultats
└── README.md            # Documentation
```

## 🛠️ Configuration

Les hyperparamètres principaux dans `trainDQN.py` :
- Taille de la grille : 10x10
- Nombre d'obstacles : 5
- Nombre d'épisodes : 2000
- Learning rate : 1e-4
- Gamma (discount factor) : 0.95
- Epsilon start/end : 1.0 → 0.05
- Batch size : 64

## 💡 Conseils de débogage

**L'agent ne progresse pas ?**
```bash
# Visualiser un épisode en détail
python visualize_agent.py --episodes 1 --delay 0.5
```

**Comparer les performances à différentes étapes ?**
- Sauvegardez des checkpoints du modèle pendant l'entraînement
- Visualisez chaque version avec `--model`

**L'agent évite-t-il les obstacles ?**
```bash
# Observer plusieurs épisodes avec des obstacles différents
python visualize_agent.py --episodes 20 --delay 0.2
```

## 📚 Documentation complète

Pour plus de détails sur la visualisation, consultez `VISUALIZATION_GUIDE.md`.

## 🤝 Auteurs

Mini-projet d'apprentissage automatique - UQAC