# Deep Learning 1 — ECG200 classification

Projet final de Deep Learning 1 : comparaison de modèles profonds pour classifier des signaux ECG normaux/anormaux sur le dataset ECG200.

## Objectif

Comparer plusieurs architectures vues en cours :

- MLP : baseline dense.
- CNN 1D : extraction de motifs locaux dans le signal ECG.
- RNN/LSTM : modèle séquentiel, à ajouter dans `src/models.py` par le collègue chargé de cette partie.

Le projet sauvegarde automatiquement les métriques, courbes d'entraînement, matrices de confusion et informations de complexité.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Lancer les expériences

Par défaut, les modèles actuellement implémentés sont entraînés : `mlp` et `cnn`.

```bash
python -m src.train
```

Pour lancer un seul modèle :

```bash
python -m src.train --models mlp
python -m src.train --models cnn
```

Pour plusieurs seeds :

```bash
python -m src.train --models mlp cnn --seeds 0 1 2 3 4
```

Quand le RNN/LSTM sera implémenté dans `src/models.py` :

```bash
python -m src.train --models mlp cnn rnn --seeds 0 1 2 3 4
```

## Résultats générés

```text
results/
├── metrics.csv
├── confusion_matrices/
└── training_curves/
```

`metrics.csv` contient notamment :

- accuracy
- precision
- recall
- f1-score
- nombre de paramètres
- temps d'entraînement
- temps d'inférence
- temps moyen d'inférence par échantillon
- taille du modèle sauvegardé

## Structure

```text
DeepLearning/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── results/
│   ├── metrics.csv
│   ├── confusion_matrices/
│   └── training_curves/
└── report/
    └── rapport.pdf
```

## Notes méthodologiques

- Les données ECG200 sont lues avec `header=None`, car les fichiers TSV n'ont pas d'en-tête.
- Le scaler est ajusté uniquement sur l'ensemble d'entraînement pour éviter toute fuite de données.
- Le split validation est stratifié pour préserver la proportion des classes.
- Le test set n'est utilisé qu'à la fin pour évaluer le modèle sélectionné sur validation.
- Les courbes train/validation permettent de discuter underfitting et overfitting dans le rapport.
