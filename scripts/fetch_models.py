"""Télécharge les checkpoints au démarrage si absents (déploiement Render).

Les .pt sont gitignorés (trop gros pour git). Hébergez-les dans une GitHub Release
et indiquez les URLs dans MODEL_URLS (séparées par des virgules).
"""
import os
import sys
import urllib.request

dest = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints")
os.makedirs(dest, exist_ok=True)
for url in filter(None, (u.strip() for u in os.environ.get("MODEL_URLS", "").split(","))):
    path = os.path.join(dest, os.path.basename(url.split("?")[0]))
    if os.path.exists(path):
        continue
    print(f"Téléchargement de {url} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # l'app démarre quand même, l'erreur s'affichera dans l'UI
        print(f"Échec du téléchargement : {exc}", file=sys.stderr)
