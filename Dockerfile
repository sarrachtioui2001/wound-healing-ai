FROM python:3.11-slim

# opencv-python (headless import path) needs libGL.so.1 and libglib at runtime,
# not just at pip-install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-deploy.txt .
# CPU-only wheels first: the default PyPI torch/torchvision wheels pull in
# CUDA dependencies (~2GB) that a CPU-only container never uses. requirements.txt
# also lists torch/torchvision unpinned, so the later install below is a no-op
# for them and only installs the remaining packages.
RUN pip install --no-cache-dir --default-timeout=180 --retries 10 \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --default-timeout=180 --retries 10 -r requirements-deploy.txt

COPY . .

# Hugging Face Spaces exécute le conteneur en utilisateur 1000 : le dossier de travail
# doit être inscriptible (résultats du suivi écrits dans outputs/).
RUN useradd -m -u 1000 user && mkdir -p outputs/checkpoints outputs/kinetics && chown -R user /app
USER user

EXPOSE 7860

# Render fournit $PORT ; en local on retombe sur 8501. Les checkpoints sont téléchargés
# au démarrage depuis MODEL_URLS s'ils sont absents (scripts/fetch_models.py).
CMD ["sh", "-c", "python scripts/fetch_models.py && streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-7860}"]
