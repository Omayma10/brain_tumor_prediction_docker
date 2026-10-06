# Brain Tumor Prediction – Docker

Application conteneurisée de prédiction de tumeur cérébrale à partir de caractéristiques extraites d'images IRM. Un modèle **Random Forest** est exposé via une **API Flask**, et une interface en ligne de commande permet de saisir les valeurs et d'obtenir la prédiction (`tumor` / `no tumor`).

> Projet à visée pédagogique : il ne constitue en aucun cas un outil de diagnostic médical.

## Architecture

Le projet repose sur deux conteneurs Docker qui communiquent via un réseau partagé (`mon-reseau`), orchestrés par Docker Compose :

```
┌────────────────────┐   POST /api/receive_values   ┌─────────────────────────┐
│  interface (cont2) │ ───────────────────────────► │  flask-server (cont1)   │
│  ihm.py (CLI)      │ ◄─────────────────────────── │  API Flask + modèle RF  │
└────────────────────┘     "tumor" / "no tumor"     └─────────────────────────┘
                                                          port 8080
```

| Conteneur | Dossier | Rôle |
|---|---|---|
| `flask-server` | `cont1/` | API Flask (port 8080) qui charge le modèle `random_forest_v2.joblib` et renvoie la prédiction |
| `interface` | `cont2/` | Interface en ligne de commande : saisie des 10 caractéristiques et envoi à l'API (avec 5 tentatives de reconnexion) |

## Structure du dépôt

```
brain_tumor_prediction_docker/
├── docker-compose.yml
├── cont1/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── brain_docker.py             # API Flask + prédiction
│   ├── random_forest_v2.joblib     # modèle entraîné
│   ├── Brain_tumor_original.csv    # jeu de données d'origine
│   └── Brain_tumor_data.csv        # exemple de format d'entrée
└── cont2/
    ├── Dockerfile
    ├── requirements.txt
    └── ihm.py                      # interface en ligne de commande
```

## Données et modèle

Le jeu de données (`Brain_tumor_original.csv`, ~3 760 images) contient des caractéristiques statistiques et de texture extraites d'images IRM, ainsi que la classe (`0` = pas de tumeur, `1` = tumeur).

Prétraitement appliqué (identique à l'entraînement) :
- suppression des colonnes `Image`, `Mean`, `Correlation` et `Coarseness` ;
- séparation train/test 80/20 (`random_state=42`) ;
- normalisation avec `MinMaxScaler` ajusté sur le jeu d'entraînement.

**10 caractéristiques d'entrée**, dans cet ordre :

`Variance`, `Standard Deviation`, `Entropy`, `Skewness`, `Kurtosis`, `Contrast`, `Energy`, `ASM`, `Homogeneity`, `Dissimilarity`

## Prérequis

- [Docker](https://docs.docker.com/get-docker/) et Docker Compose

## Lancement

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/Omayma10/brain_tumor_prediction_docker.git
   cd brain_tumor_prediction_docker
   ```

2. Construire et démarrer les conteneurs :
   ```bash
   docker compose up --build
   ```

3. Dans un second terminal, s'attacher à l'interface pour saisir les valeurs :
   ```bash
   docker compose ps                  # repérer le nom du conteneur « interface »
   docker attach <nom_du_conteneur_interface>
   ```
   L'interface attend 10 secondes au démarrage (le temps que l'API soit prête), puis demande chaque valeur :
   ```
   Variance:
   Standard Deviation:
   Entropy:
   ...
   Dissimilarity:
   ```
   La réponse s'affiche : `tumor` ou `no tumor`.

4. Pour tout arrêter :
   ```bash
   docker compose down
   ```

> Sous Windows, si le build échoue à cause des chemins `.\cont1` / `.\cont2` dans `docker-compose.yml`, utiliser `./cont1` et `./cont2`, qui fonctionnent sur tous les systèmes.

## Utilisation directe de l'API

L'API est aussi accessible depuis la machine hôte sur le port 8080 :

```bash
curl -X POST http://localhost:8080/api/receive_values \
  -H "Content-Type: application/json" \
  -d '{"values": [1500.0, 38.7, 0.08, 4.2, 25.1, 180.5, 0.35, 0.12, 0.62, 5.4]}'
```

Réponse : `tumor` ou `no tumor` (ou un JSON `{"success": false, "message": ...}` en cas de valeurs invalides ou d'erreur).

## Technologies

- Python 3.10
- scikit-learn 1.2.2 (Random Forest), pandas, joblib
- Flask
- Docker & Docker Compose

## Auteure

**Omayma Raji** – Etudiante EPISEN
