# Mise en production — Classification ECG200

## 1. Contenu du dossier

Structure :

```text
prod/
├── ia/
│   ├── Dockerfile
│   ├── inference.py
│   ├── main.py
│   └── requirements.txt
├── models/
│   ├── ecg_model.keras
│   └── preprocess.json
├── nn/
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/
│       ├── java/fr/uha/ecg/production/
│       │   ├── EcgController.java
│       │   └── EcgFrontApplication.java
│       └── resources/
│           ├── application.properties
│           └── static/index.html
├── docker-compose.yml
├── go
├── .dockerignore
└── README_PRODUCTION.md
```

Les deux fichiers relatifs à l IA utilisée sont obligatoires :

```text
models/ecg_model.keras
models/preprocess.json
```

Sans ces fichiers, la mise en production ne peut pas fonctionner : ce module consomme les artefacts produits par le module d'entraînement, mais ne les génère pas.

## 2. Architecture

Le déploiement lance deux conteneurs Docker :

```text
navigateur → nn Spring Boot → ia Flask/TensorFlow → modèle Keras
```

### Service `ia`

Le service `ia` est le backend d'inférence. Il :

- charge `models/ecg_model.keras` ;
- charge `models/preprocess.json` ;
- vérifie que chaque signal contient exactement 96 valeurs ;
- applique le prétraitement défini dans `preprocess.json` ;
- adapte la forme d'entrée au modèle Keras ;
- renvoie la classe prédite, la confiance et les probabilités par classe.

Ce service écoute sur le port `80` à l'intérieur du réseau Docker, mais il n'est pas exposé directement sur la machine hôte.

### Service `nn`

Le service `nn` est le frontal Spring Boot. Il :

- sert l'interface web ;
- expose les routes `/api/...` ;
- relaie les requêtes vers le backend `ia`.

Le frontal est exposé sur la machine hôte à l'adresse :

```text
http://localhost:8080
```

## 3. Prérequis

Il faut avoir Docker et Docker Compose installés.

## 4. Lancement sous Linux

Depuis le dossier `prod/` :

```bash
chmod +x go
./go
```

Le script `go` vérifie d'abord que les artefacts obligatoires existent :

```text
models/ecg_model.keras
models/preprocess.json
```

Puis il lance :

```bash
docker compose up --build
```

## 5. Lancement sous Windows PowerShell + Docker Desktop

Depuis le dossier `prod/` :

```powershell
docker compose up --build
```

Le script `go` est un script bash. Il est prévu pour Linux/macOS/WSL, pas pour PowerShell classique.

## 6. Utilisation de l'interface web

Une fois les conteneurs lancés, ouvrir :

```text
http://localhost:8080
```

L'interface permet de :

- coller un signal ECG de 96 valeurs ;
- vérifier l'état du service IA ;
- charger/configurer le modèle ;
- classifier le signal.

Le format attendu est une suite de 96 valeurs numériques séparées par des espaces, virgules, points-virgules ou retours ligne.

Exemple de format :

```text
0.000000 0.014237 0.028184 0.041557 ... 0.054508
```

## 7. Utilisation de l'API

Toutes les routes publiques passent par le frontal Spring Boot.

### Vérifier le service IA

```bash
curl http://localhost:8080/api/health
```

### Charger et vérifier le modèle

```bash
curl -X POST http://localhost:8080/api/config
```

La réponse doit indiquer notamment :

```json
{
  "model_loaded": true,
  "preprocess_exists": true,
  "preprocess_mode": "standard_scaler"
}
```

### Classifier un signal

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{"signal":[0.0,0.014237,0.028184,0.041557,0.054083,0.065508,0.075598,0.084147,0.090982,0.095964,0.098990,0.100000,0.098972,0.095928,0.090930,0.084079,0.075515,0.065412,0.053977,0.041442,0.028063,0.014112,-0.000126,-0.014362,-0.028306,-0.041672,-0.054190,-0.065603,-0.075680,-0.084215,-0.091035,-0.095999,-0.099008,-0.100000,-0.098954,-0.095892,-0.090877,-0.084010,-0.075432,-0.065317,-0.053871,-0.041327,-0.027942,-0.013987,0.000253,0.014487,0.028427,0.041787,0.054296,0.065699,0.075763,0.084283,0.091087,0.096035,0.099026,0.100000,0.098936,0.095856,0.090824,0.083942,0.075349,0.065221,0.053764,0.041212,0.027820,0.013862,-0.000379,-0.014613,-0.028548,-0.041902,-0.054402,-0.065794,-0.075845,-0.084351,-0.091139,-0.096070,-0.099043,-0.099999,-0.098917,-0.095820,-0.090771,-0.083873,-0.075265,-0.065125,-0.053657,-0.041097,-0.027699,-0.013736,0.000506,0.014738,0.028669,0.042017,0.054508]}'
```

Réponse typique :

```json
{
  "status": "ok",
  "data": {
    "predicted_class_name": "-1",
    "predicted_display_label": "Normal heartbeat",
    "confidence": 0.91,
    "probability_by_class": {
      "-1": 0.91,
      "1": 0.09
    }
  }
}
```

### Classifier plusieurs signaux

```bash
curl -X POST http://localhost:8080/api/batch \
  -H "Content-Type: application/json" \
  -d '{"signals":[[0.0,0.014237,0.028184,0.041557,0.054083,0.065508,0.075598,0.084147,0.090982,0.095964,0.098990,0.100000,0.098972,0.095928,0.090930,0.084079,0.075515,0.065412,0.053977,0.041442,0.028063,0.014112,-0.000126,-0.014362,-0.028306,-0.041672,-0.054190,-0.065603,-0.075680,-0.084215,-0.091035,-0.095999,-0.099008,-0.100000,-0.098954,-0.095892,-0.090877,-0.084010,-0.075432,-0.065317,-0.053871,-0.041327,-0.027942,-0.013987,0.000253,0.014487,0.028427,0.041787,0.054296,0.065699,0.075763,0.084283,0.091087,0.096035,0.099026,0.100000,0.098936,0.095856,0.090824,0.083942,0.075349,0.065221,0.053764,0.041212,0.027820,0.013862,-0.000379,-0.014613,-0.028548,-0.041902,-0.054402,-0.065794,-0.075845,-0.084351,-0.091139,-0.096070,-0.099043,-0.099999,-0.098917,-0.095820,-0.090771,-0.083873,-0.075265,-0.065125,-0.053657,-0.041097,-0.027699,-0.013736,0.000506,0.014738,0.028669,0.042017,0.054508]]}'
```

## 8. Format des artefacts

### `models/ecg_model.keras`

Modèle Keras déjà entraîné par le module d'entraînement.

Dans cette version, le modèle attendu est un classifieur ECG200 binaire.

### `models/preprocess.json`

Fichier décrivant le prétraitement à appliquer avant l'inférence.

Format attendu :

```json
{
  "input_length": 96,
  "class_names": ["-1", "1"],
  "display_labels": {
    "-1": "Normal heartbeat",
    "1": "Myocardial infarction / abnormal heartbeat"
  },
  "scaler": {
    "mean": [96 valeurs],
    "scale": [96 valeurs]
  }
}
```

Le mode par défaut est :

```text
PREPROCESS_MODE=standard_scaler
```

Donc le backend applique :

```text
x_normalized = (x - mean) / scale
```

Le fichier `preprocess.json` doit provenir du même entraînement que le modèle choisi, ou être recalculé avec exactement le même split d'entraînement.

## 9. Variables d'environnement

Les principales variables utilisées par le conteneur `ia` sont :

| Variable | Valeur par défaut | Description |
|---|---:|---|
| `MODEL_PATH` | `/app/models/ecg_model.keras` | Chemin du modèle Keras dans le conteneur |
| `PREPROCESS_PATH` | `/app/models/preprocess.json` | Chemin du fichier de prétraitement |
| `PREPROCESS_MODE` | `standard_scaler` | Mode de prétraitement |
| `INPUT_LENGTH` | `96` | Nombre de valeurs attendues par signal |

Ces valeurs sont définies dans `docker-compose.yml`.

## 10. Arrêt des conteneurs

Depuis le dossier `prod/` :

```bash
docker compose down
```

Pour supprimer aussi les images construites localement :

```bash
docker compose down --rmi local
```

## 11. Problèmes fréquents

### `models/ecg_model.keras is missing`

Le modèle entraîné n'est pas présent.

Vérifier :

```bash
ls -lh models/ecg_model.keras
```

### `models/preprocess.json is missing`

Le fichier de prétraitement n'est pas présent.

Vérifier :

```bash
ls -lh models/preprocess.json
```

### Le navigateur n'arrive pas à ouvrir `localhost`

Utiliser le port `8080` :

```text
http://localhost:8080
```

`localhost` sans port tente le port `80`, qui n'est pas exposé par le frontal.

### `Expected 96 ECG values`

Le signal envoyé ne contient pas exactement 96 valeurs numériques.

### Docker ne répond pas

Vérifier que le daemon Docker tourne :

```bash
docker ps
```

Sur Linux :

```bash
sudo systemctl status docker
```

Sur Windows, Docker Desktop doit être lancé.

## 12. Limites du module

Ce dossier ne fait pas :

- l'entraînement des modèles ;
- la comparaison MLP/CNN/RNN ;
- le calcul des métriques ;
- la génération des courbes ;
- la sélection automatique du meilleur modèle.

Ces étapes appartiennent au module d'entraînement. Ce dossier correspond uniquement à la mise en production.
