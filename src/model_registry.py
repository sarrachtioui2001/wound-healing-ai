"""Enregistrement du meilleur checkpoint d'un run dans le MLflow Model Registry, et
promotion automatique vers l'alias "production" si son Dice de validation dépasse
celui de la version actuellement en production (ou si aucune version n'y est encore).

Utilise l'API d'alias de MLflow (set_registered_model_alias / get_model_version_by_alias),
pas les "stages" (Staging/Production) dépréciés depuis MLflow 2.9.

La décision de promotion (doit_promouvoir_en_production) est une fonction pure, testée
sans dépendance à un serveur MLflow réel -- seule enregistrer_et_promouvoir parle au
tracking server.
"""

REGISTERED_MODEL_NAME = "wound-segmentation-unet"
PRODUCTION_ALIAS = "production"


def doit_promouvoir_en_production(nouveau_dice: float, dice_production_actuel: float | None) -> bool:
    """Décide si une nouvelle version doit devenir la version "production" : oui si
    aucune version n'y est encore, ou si son Dice de validation est strictement
    meilleur que celui de la version actuellement en production. À égalité, la
    version en production actuelle est conservée (pas de remplacement sans gain réel)."""
    return dice_production_actuel is None or nouveau_dice > dice_production_actuel


def dice_production_actuel(client, name: str = REGISTERED_MODEL_NAME) -> float | None:
    """Dice de validation (métrique "best_val_dice" du run source) de la version
    actuellement pointée par l'alias "production", ou None si l'alias n'existe pas
    encore (premier enregistrement, ou modèle jamais promu)."""
    try:
        version = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except Exception:
        return None
    run = client.get_run(version.run_id)
    return run.data.metrics.get("best_val_dice")


def enregistrer_et_promouvoir(
    run_id: str, artifact_path: str, val_dice: float, name: str = REGISTERED_MODEL_NAME
) -> str:
    """Enregistre l'artefact du run comme nouvelle version de `name`, et la promeut
    en "production" si elle bat la version actuelle (voir doit_promouvoir_en_production).

    run_id: id du run MLflow (run.info.run_id) qui a déjà loggé l'artefact.
    artifact_path: chemin de l'artefact relatif au run (ex: nom du fichier .pt).
    val_dice: best_val_dice du run, pour la comparaison de promotion.

    Retourne un message court décrivant l'issue, destiné à être loggé par l'appelant."""
    import mlflow  # import tardif : mlflow absent du déploiement (app seule)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    version = mlflow.register_model(model_uri, name)

    dice_actuel = dice_production_actuel(client, name)
    if doit_promouvoir_en_production(val_dice, dice_actuel):
        client.set_registered_model_alias(name, PRODUCTION_ALIAS, version.version)
        reference = f"{dice_actuel:.4f}" if dice_actuel is not None else "aucune"
        return (
            f"version {version.version} de {name} promue en production "
            f"(Dice {val_dice:.4f} > production actuelle : {reference})"
        )
    return (
        f"version {version.version} de {name} enregistrée, non promue "
        f"(Dice {val_dice:.4f} <= production actuelle {dice_actuel:.4f})"
    )
