# Mise en production — ECG200

Ce dossier est un module de mise en production uniquement.

Il est indépendant du projet d'entraînement au moment du rendu, mais il ne réentraîne aucun modèle. Il attend des artefacts déjà produits par le module d'entraînement.

## Contenu

```text
ecg_mise_en_prod/
├── docker-compose.yml
├── go
├── ia/          # API Flask/TensorFlow privée
├── nn/          # frontal Spring Boot public sur localhost:8080
├── models/      # artefacts entraînés à déposer ici
└── samples/     # exemple de requête JSON
```

Architecture Docker :

- `ia` charge le modèle Keras et réalise l'inférence ;
- `nn` expose l'interface HTTP sur `http://localhost:8080` ;
- seul `nn` expose un port vers l'hôte ;
- `ia` reste accessible uniquement dans le réseau Docker interne.

## Artefacts obligatoires

Avant de lancer la production, placer dans `models/` :

```text
models/ecg_model.keras
models/preprocess.json
```

`models/ecg_model.keras` doit être un modèle déjà entraîné par le module d'entraînement, par exemple un fichier issu de :

```text
results/models/<run_selectionne>.keras
```

`models/preprocess.json` doit contenir les paramètres de prétraitement associés au modèle. Par défaut, le service attend un `StandardScaler` identique à celui utilisé à l'entraînement.

Le module de production ne crée pas ces fichiers. S'ils sont absents, le service ne peut pas fonctionner correctement.

Format attendu de `preprocess.json` :

```json
{
  "input_length": 96,
  "class_names": ["-1", "1"],
  "display_labels": {
    "-1": "class -1",
    "1": "class 1"
  },
  "scaler": {
    "mean": [96 valeurs],
    "scale": [96 valeurs]
  }
}
```

## Lancement Linux

Depuis le dossier `ecg_mise_en_prod/` :

```bash
chmod +x go
./go
```

Équivalent direct :

```bash
docker compose up --build
```

## Lancement PowerShell

Depuis le dossier `ecg_mise_en_prod/` :

```powershell
docker compose up --build
```

## Interface

```text
http://localhost:8080
```

## Tests API

Linux/macOS :

```bash
curl http://localhost:8080/api/health
curl -X POST http://localhost:8080/api/config
curl -X POST http://localhost:8080/api/predict \
  -H 'Content-Type: application/json' \
  --data @samples/ecg_example.json
```

PowerShell :

```powershell
Invoke-RestMethod http://localhost:8080/api/health
Invoke-RestMethod -Method Post http://localhost:8080/api/config
Invoke-RestMethod -Method Post http://localhost:8080/api/predict -ContentType 'application/json' -InFile samples/ecg_example.json
```

## Variables de configuration

| Variable | Défaut | Rôle |
|---|---:|---|
| `MODEL_PATH` | `/app/models/ecg_model.keras` | modèle Keras chargé par le service IA |
| `PREPROCESS_PATH` | `/app/models/preprocess.json` | statistiques du prétraitement |
| `PREPROCESS_MODE` | `standard_scaler` | normalisation appliquée avant prédiction |
| `INPUT_LENGTH` | `96` | nombre de valeurs attendues dans un signal ECG200 |

Modes supportés :

- `standard_scaler` : mode recommandé, nécessite `preprocess.json` ;
- `none` : aucune normalisation, à utiliser seulement si le modèle entraîné attend déjà des entrées normalisées ailleurs ;
- `per_sample_zscore` : normalisation par échantillon, uniquement si le modèle a été entraîné comme cela.
