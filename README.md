
# GNN Intrusion Detection System (IDS)

Ce projet utilise les **Réseaux de Neurones sur Graphes (GNN)** pour la détection d'intrusions réseau. L'objectif est de classifier le trafic réseau en deux catégories : **Normal** ou **Attack** (Attaque), en se basant sur un dataset contenant 41 caractéristiques (features) par nœud.

Le projet compare les performances de deux architectures GNN populaires : **GAT** (Graph Attention Network) et **GraphSAGE**.

---

## Dataset
- **Nombre total de nœuds (exemples) :** 494,021
- **Nombre de caractéristiques (features) :** 41 (Type KDD99/NSL-KDD)
- **Classes :** `normal` (14,493 tests) vs `attack` (59,611 tests)
- **Tâche :** Classification Binaire de Nœuds (Node Classification).

---

## Architectures des Modèles

### 1. Modèle GAT (Graph Attention Network)
- **Architecture :** GATConv (128 unités) ➔ ELU ➔ GATConv (2 unités, sortie)
- **Entraînement :** 100 Epochs
- **Loss Finale :** `0.0995`

### 2. Modèle GraphSAGE
- **Architecture :** SAGEConv (64 unités) ➔ ReLU ➔ Dropout ➔ SAGEConv (2 unités, sortie)
- **Entraînement :** 100 Epochs
- **Loss Finale :** `0.0076`

---

## Résultats et Évaluation (Test Set: 74,104 exemples)

Les deux modèles ont montré des performances exceptionnelles sur l'ensemble de test, avec un léger avantage pour GraphSAGE en termes d'accuracy globale et de convergence (Loss plus faible).

| Métrique | GAT (Graph Attention Network) | GraphSAGE |
| :--- | :---: | :---: |
| **Accuracy** | **99.71 %** | **99.84 %** |
| **Précision (Normal/Attack)** | 0.99 / 1.00 | 0.99 / 1.00 |
| **Recall (Normal/Attack)** | 1.00 / 1.00 | 1.00 / 1.00 |
| **F1-Score Macro** | 1.00 | 1.00 |

### Rapport de Classification (GraphSAGE) :

              precision    recall  f1-score   support

      normal       0.99      1.00      1.00     14493
      attack       1.00      1.00      1.00     59611

    accuracy                           1.00     74104
   macro avg       1.00      1.00      1.00     74104
weighted avg       1.00      1.00      1.00     74104

Technologies Utilisées

  - Python 3
  - PyTorch & PyTorch Geometric (PyG) (Pour la modélisation GNN)
  - Scikit-Learn (Pour l'évaluation des performances)
  - Google Colab (Environnement de développement et d'entraînement)

Comment utiliser ce projet ?

Ce projet a été développé sur Google Colab. Pour l'exécuter :

1.  Clonez ce repository :
    git clone https://github.com/Kellaikram/GNN-Intrusion-Detection.git
2.  Ouvrez le fichier .ipynb dans Google Colab ou Jupyter Notebook.
3.  Assurez-vous d'installer PyTorch Geometric avant de lancer les cellules :
    !pip install torch_geometric
4.  Exécutez le notebook cellule par cellule.
