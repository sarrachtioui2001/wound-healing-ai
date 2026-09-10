import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))
from chatbot import corriger_references_ordinales, detecter_citations_suspectes

SOURCES = [
    "Studying Cell Migration (Random and Wound Healing) Parameters with Imaging and MATLAB Analysis",
    "Study of Wound Healing Dynamics by Single Pseudo-Particle Tracking in Phase Contrast Images Acquired in Time-Lapse",
    "Migratory Metrics of Wound Healing: A Quantification Approach for in vitro Scratch Assays",
]


def test_titre_invente_absent_des_sources_est_detecte():
    # Cas réel observé (2026-09-08) : titre plausible mais absent du corpus.
    texte = (
        'Un article intitulé "Analysis of wound healing using digital image analysis" '
        "explique les méthodes de traitement de l'image."
    )
    suspectes = detecter_citations_suspectes(texte, SOURCES)
    assert suspectes == ["Analysis of wound healing using digital image analysis"]


def test_titre_recopie_exactement_n_est_pas_signale():
    texte = f'D\'après "{SOURCES[0]}", la migration cellulaire est mesurable par imagerie.'
    assert detecter_citations_suspectes(texte, SOURCES) == []


def test_titre_recopie_avec_diacritique_different_n_est_pas_signale():
    # Variante unicode plausible (apostrophe/tiret différent) : ne doit pas être
    # signalée comme fabriquée juste à cause d'un caractère de ponctuation.
    texte = 'Voir "Migratory Metrics of Wound Healing: A Quantification Approach for in-vitro Scratch Assays".'
    assert detecter_citations_suspectes(texte, SOURCES) == []


def test_aucune_citation_ne_renvoie_liste_vide():
    texte = "Ce texte ne cite aucun article entre guillemets, juste des chiffres : 39,8 %."
    assert detecter_citations_suspectes(texte, SOURCES) == []


def test_citation_courte_ignoree_pas_un_titre():
    # Emphase courte entre guillemets ("hétérogénéité"), pas une tentative de
    # citation d'article -- ne doit pas déclencher un faux positif.
    texte = 'Cette valeur cache une certaine "hétérogénéité" entre intervalles.'
    assert detecter_citations_suspectes(texte, SOURCES) == []


def test_aucune_source_disponible_signale_toute_citation_longue():
    texte = 'Voir "Un article totalement hors de propos et jamais fourni ici".'
    assert detecter_citations_suspectes(texte, []) == [
        "Un article totalement hors de propos et jamais fourni ici"
    ]


def test_correction_references_ordinales_remplace_les_trois_positions():
    # Cas réel observé (2026-09-10, retour biologiste) : le LLM cite par position
    # malgré l'interdiction explicite du prompt.
    texte = (
        "Le premier article établit une méthode. "
        "Le second article décrit un cadre d'analyse. "
        "Le troisième article propose une méthode fiable."
    )
    resultat = corriger_references_ordinales(texte, SOURCES)
    assert f"« {SOURCES[0]} »" in resultat
    assert f"« {SOURCES[1]} »" in resultat
    assert f"« {SOURCES[2]} »" in resultat
    assert "premier article" not in resultat
    assert "second article" not in resultat
    assert "troisième article" not in resultat


def test_correction_references_ordinales_preserve_la_majuscule_en_debut_de_phrase():
    resultat = corriger_references_ordinales("Le premier article établit...", SOURCES)
    assert resultat.startswith("L'article")


def test_correction_references_ordinales_variante_deuxieme_et_minuscule():
    # "deuxième" (pas seulement "second") et une occurrence en milieu de phrase
    # (minuscule) doivent aussi être corrigées.
    texte = "Selon le deuxième article, la migration collective est mesurable."
    resultat = corriger_references_ordinales(texte, SOURCES)
    assert f"l'article « {SOURCES[1]} »" in resultat


def test_correction_references_ordinales_index_hors_limite_laisse_intact():
    # "le quatrième article" alors qu'il n'y a que 3 sources : mieux vaut laisser
    # la référence telle quelle qu'un remplacement incorrect.
    texte = "Le quatrième article apporte un éclairage supplémentaire."
    assert corriger_references_ordinales(texte, SOURCES) == texte


def test_correction_references_ordinales_texte_sans_reference_est_inchange():
    texte = "Ce texte ne fait aucune référence par position à un article."
    assert corriger_references_ordinales(texte, SOURCES) == texte
