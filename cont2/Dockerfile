# Utilise une image Python officielle
FROM python:3.10-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie le code source dans le conteneur
COPY ihm.py .

# Installe les dépendances Python
RUN pip install --no-cache-dir requests

# Commande par défaut pour exécuter l'application
CMD ["python", "ihm.py"]

