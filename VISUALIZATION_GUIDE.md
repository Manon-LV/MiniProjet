# 📊 Visualisation de l'Agent DQN dans GridWorld

Ce guide explique comment utiliser la visualisation en temps réel pour observer votre agent DQN naviguer dans l'environnement GridWorld.

## 🎯 Fonctionnalités

La visualisation affiche :
- **🔵 Agent** : Position actuelle de l'agent (cercle bleu)
- **⭐ Objectif** : Position de la cible (étoile verte)
- **⬛ Obstacles** : Obstacles dans la grille (carrés gris foncé)
- **📈 Métriques** : Numéro d'épisode, nombre de steps, reward cumulé

## 🚀 Utilisation

### 1️⃣ Visualisation pendant l'entraînement

Pour activer la visualisation pendant l'entraînement :

```bash
python trainDQN.py --visualize
```

Options disponibles :
- `--visualize` : Active la visualisation
- `--render-interval N` : Visualise tous les N épisodes (défaut: 100)

**Exemples :**

```bash
# Visualiser tous les 50 épisodes
python trainDQN.py --visualize --render-interval 50

# Visualiser tous les 10 épisodes (plus fréquent)
python trainDQN.py --visualize --render-interval 10
```

⚠️ **Note** : La visualisation ralentit l'entraînement. Utilisez des intervalles larges (100-500).

### 2️⃣ Visualisation d'un agent entraîné

Pour observer un agent déjà entraîné :

```bash
python visualize_agent.py
```

Options disponibles :
- `--model PATH` : Chemin vers le modèle (défaut: `dqn_model.pth`)
- `--episodes N` : Nombre d'épisodes à visualiser (défaut: 5)
- `--delay SEC` : Délai entre les frames en secondes (défaut: 0.3)
- `--size N` : Taille de la grille (défaut: 10)
- `--obstacles N` : Nombre d'obstacles (défaut: 5)

**Exemples :**

```bash
# Visualiser 10 épisodes avec une vitesse plus lente
python visualize_agent.py --episodes 10 --delay 0.5

# Visualiser sur une grille plus grande
python visualize_agent.py --size 15 --obstacles 10

# Utiliser un modèle spécifique
python visualize_agent.py --model mon_modele.pth --episodes 20
```

## 📋 Interprétation des métriques

### Pendant la visualisation :
- **Steps** : Nombre d'actions effectuées (max: 200 par défaut)
- **Reward** : Récompense cumulée de l'épisode
  - `+10` : Atteindre l'objectif
  - `-5` : Heurter un obstacle
  - `-0.01` : Chaque step
  - `-0.5` : Timeout (max_step atteint)

### Après les épisodes :
- **Taux de succès** : Pourcentage d'épisodes où l'agent atteint l'objectif
- **Reward moyen** : Performance moyenne de l'agent
- **Steps moyen** : Efficacité du chemin trouvé

## 🎨 Légende des couleurs

- 🔵 **Bleu** : Agent (votre IA)
- 🟢 **Vert** : Objectif/Cible
- ⬛ **Gris foncé** : Obstacles
- ⬜ **Blanc** : Cases vides

## 💡 Conseils d'utilisation

### Pour déboguer votre agent :

1. **Agent qui tourne en rond ?**
   ```bash
   python visualize_agent.py --delay 0.1 --episodes 3
   ```
   → Observez si l'agent explore ou reste bloqué

2. **Agent qui fonce dans les obstacles ?**
   ```bash
   python visualize_agent.py --delay 0.5 --episodes 10
   ```
   → Vérifiez si l'agent apprend à éviter les obstacles

3. **Comparer début vs fin d'entraînement :**
   - Sauvegardez le modèle à différents moments
   - Visualisez chaque version avec `--model`

### Pour optimiser la vitesse :

- **Rapide** : `--delay 0.01` (bon pour vérifier rapidement)
- **Normal** : `--delay 0.1` à `0.3` (bon équilibre)
- **Lent** : `--delay 0.5` à `1.0` (analyse détaillée)

## 🛠️ Dépendances

Assurez-vous d'avoir installé :
```bash
pip install torch numpy matplotlib
```

## 🐛 Résolution de problèmes

**Problème** : La fenêtre ne s'affiche pas
- **Solution** : Vérifiez que vous n'êtes pas en SSH sans X11 forwarding

**Problème** : "Model not found"
- **Solution** : Entraînez d'abord avec `python trainDQN.py`
- Le modèle sera sauvegardé automatiquement dans `dqn_model.pth`

**Problème** : Visualisation trop lente/rapide
- **Solution** : Ajustez `--delay` (plus petit = plus rapide)

**Problème** : Fenêtre qui reste ouverte
- **Solution** : Appuyez sur `Ctrl+C` dans le terminal

## 📚 Structure du code

```
GridWorld.py
├── render()        # Visualise l'état actuel
└── close()         # Ferme la fenêtre

trainDQN.py         # Entraînement avec option --visualize
visualize_agent.py  # Script dédié à la visualisation
```

## 🎯 Exemple de workflow complet

```bash
# 1. Entraîner avec visualisation occasionnelle
python trainDQN.py --visualize --render-interval 200

# 2. Visualiser l'agent entraîné en détail
python visualize_agent.py --episodes 20 --delay 0.2

# 3. Tester sur des configurations différentes
python visualize_agent.py --size 15 --obstacles 15 --episodes 10
```

Bon debug ! 🚀
