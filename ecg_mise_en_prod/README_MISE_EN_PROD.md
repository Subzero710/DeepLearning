# Mise en production — ECG200 DeepLearning

Structure conforme au TD `dockerisation` :

- `ia/` : conteneur Python/Flask qui charge le modèle TensorFlow/Keras ;
- `nn/` : frontal Spring Boot exposé sur `localhost:8080` ;
- `docker-compose.yml` : réseau Docker interne, seul le frontal publie un port hôte.

Le service IA est caché : il n'expose pas de port sur l'hôte et le frontal l'appelle via `http://ia:80`.

## Adaptation au dépôt GitHub

Le dépôt `Subzero710/DeepLearning` entraîne les modèles avec `src.train` et sauvegarde les checkpoints Keras dans `results/models/{model}_seed{seed}.keras`.

Le dépôt normalise ECG200 avec un `StandardScaler` ajusté uniquement sur le train split. La prod doit donc réutiliser les mêmes statistiques. Le script `scripts/export_production_artifacts.py` copie le modèle retenu vers `models/ecg_model.keras` et génère `models/preprocess.json`.

Les modèles MLP, CNN et RNN/LSTM sont supportés. L'API détecte automatiquement si le modèle attend `(1, 96)` ou `(1, 96, 1)`. Les fichiers `projet_*.py` ne sont pas utilisés.

## Commandes

Depuis la racine du dépôt après l'entraînement :

```bash
python ecg_mise_en_prod/scripts/export_production_artifacts.py \
  --metrics results/metrics.csv \
  --out-dir ecg_mise_en_prod/models
```

Le script choisit automatiquement le meilleur modèle selon `f1_macro`, puis `accuracy`.

Pour forcer un modèle :

```bash
python ecg_mise_en_prod/scripts/export_production_artifacts.py \
  --model results/models/rnn_seed0.keras \
  --out-dir ecg_mise_en_prod/models
```

Le seed est inféré depuis `_seed0`. S'il n'est pas dans le nom, ajouter `--seed 0`.

Lancement :

```bash
cd ecg_mise_en_prod
chmod +x go
./go
```

Interface :

```text
http://localhost:8080
```

Tests :

```bash
curl http://localhost:8080/api/health
curl -X POST http://localhost:8080/api/config
curl -X POST http://localhost:8080/api/predict \
  -H 'Content-Type: application/json' \
  --data @samples/ecg_example.json
```

## Artefacts requis

```text
models/ecg_model.keras
models/preprocess.json
```

Sans `preprocess.json`, l'IA refuse de prédire en mode `PREPROCESS_MODE=standard_scaler`, car les prédictions ne reproduiraient pas le pipeline d'entraînement.

## Variables

| Variable | Défaut | Rôle |
|---|---:|---|
| `MODEL_PATH` | `/app/models/ecg_model.keras` | modèle Keras chargé par l'IA |
| `PREPROCESS_PATH` | `/app/models/preprocess.json` | statistiques du scaler |
| `PREPROCESS_MODE` | `standard_scaler` | normalisation identique au dépôt |
| `INPUT_LENGTH` | `96` | longueur d'un signal ECG200 |

Modes disponibles : `standard_scaler`, `none`, `per_sample_zscore`.

## Préchargement du TD

Depuis le zip `dockerisation` du TD :

```bash
chmod +x preload
./preload -e
```

`-e` installe Docker Engine et le plugin Compose. Maven et Java sont utilisés dans le conteneur `nn`, pas sur l'hôte.
