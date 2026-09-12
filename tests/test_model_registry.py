"""Teste la décision de promotion du Model Registry (doit_promouvoir_en_production),
une fonction pure sans dépendance à un serveur MLflow réel. Les fonctions qui parlent
réellement au tracking server (dice_production_actuel, enregistrer_et_promouvoir) ne
sont pas testées ici -- elles nécessiteraient un serveur MLflow, hors scope de tests
unitaires rapides (cohérent avec le reste du projet : voir test_chatbot_llm_regression.py
pour le seul test dépendant d'un service externe, explicitement isolé et auto-ignoré)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_registry import doit_promouvoir_en_production


def test_promeut_si_aucune_version_en_production():
    assert doit_promouvoir_en_production(0.85, None) is True


def test_promeut_si_dice_strictement_meilleur():
    assert doit_promouvoir_en_production(0.90, 0.85) is True


def test_ne_promeut_pas_si_dice_inferieur():
    assert doit_promouvoir_en_production(0.80, 0.85) is False


def test_ne_promeut_pas_a_egalite():
    # À égalité, la version en production actuelle est conservée (pas de remplacement
    # sans gain réel) -- évite de "promouvoir" indéfiniment des versions équivalentes.
    assert doit_promouvoir_en_production(0.85, 0.85) is False
