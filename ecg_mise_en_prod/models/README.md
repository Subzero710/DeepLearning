# Modèle et prétraitement

Ce dossier doit contenir les deux artefacts de production :

```text
ecg_model.keras
preprocess.json
```

`ecg_model.keras` est le meilleur modèle sauvegardé par `src.train` dans `results/models/`.

`preprocess.json` contient les statistiques du `StandardScaler` utilisées pendant l'entraînement. Il est généré par :

```bash
python scripts/export_production_artifacts.py --out-dir models
```

Sans `preprocess.json`, le conteneur IA refuse de prédire en mode `PREPROCESS_MODE=standard_scaler`, car le projet entraîne les modèles avec une normalisation ajustée sur le train split.
