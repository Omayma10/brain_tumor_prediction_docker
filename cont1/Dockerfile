# Utilise une image de base Python plus récente.
FROM python:3.10-slim-buster

# Définit le répertoire de travail dans le conteneur.
WORKDIR /app

# Copie le fichier des dépendances et installe les dépendances.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie les fichiers de l'application et les données dans le conteneur.
COPY brain_docker.py .
COPY Brain_tumor_data.csv .
COPY Brain_tumor_original.csv .
COPY random_forest_v2.joblib .

# Commande à exécuter lors du démarrage du conteneur.
CMD ["python", "./brain_docker.py"]
