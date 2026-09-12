"""Vérifie que les chiffres injectés dans le prompt du résumé scientifique
(synthèse globale + vitesses par intervalle) correspondent aux mesures fournies,
sans jamais dépendre du LLM. Couvre la régression corrigée dans chatbot.py :
le LLM inventait des pourcentages/vitesses différents des mesures réelles quand
on le laissait les calculer lui-même."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))
from chatbot import (
    _calculer_intervalles,
    _calculer_synthese,
    _format_intervalles,
    _format_synthese,
    _phrase_heterogeneite,
    _phrase_interpretation,
    _phrase_reouverture,
    corriger_accord_conditions_biologiques,
    detecter_contradiction_reouverture,
    limiter_paragraphe_contexte,
    nettoyer_placeholder_reouverture,
)

ROWS = [
    {"time_h": 0, "area_px2": 8410, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 5063, "closure_pct": 40.0},
    {"time_h": 48, "area_px2": 2407, "closure_pct": 71.6},
]

# Cas réel signalé par le biologiste (2026-09-09) : la plaie se referme puis se
# rouvre partiellement -- delta_closure_pct négatif sur le 2e intervalle.
ROWS_REOUVERTURE = [
    {"time_h": 0, "area_px2": 94229, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 73791, "closure_pct": 21.7},
    {"time_h": 48, "area_px2": 92521, "closure_pct": 1.8},
]

# Mêmes mesures que ROWS_REOUVERTURE mais dans l'ordre chronologique correct
# (sans réouverture) : les deux intervalles sont positifs (0,07%/h puis 0,83%/h).
ROWS_SANS_REOUVERTURE = [
    {"time_h": 0, "area_px2": 94229, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 92521, "closure_pct": 1.8},
    {"time_h": 48, "area_px2": 73791, "closure_pct": 21.7},
]


def test_synthese_reprend_les_valeurs_finales_sans_les_recalculer():
    synthese = _calculer_synthese(ROWS)
    assert synthese["t0"]["area_px2"] == 8410
    assert synthese["t_final"]["area_px2"] == 2407
    assert synthese["fermeture_finale_pct"] == 71.6
    assert synthese["duree_h"] == 48
    assert synthese["vitesse_moyenne_pct_h"] == pytest.approx(1.49, abs=0.01)


def test_format_synthese_contient_les_bonnes_valeurs():
    texte = _format_synthese(_calculer_synthese(ROWS), n_mesures=len(ROWS))
    assert "8410" in texte
    assert "71,6%" in texte
    assert "1,49%/h" in texte
    assert "3" in texte  # nombre de mesures rappelé pour éviter de sur-interpréter la tendance


def test_intervalles_detectent_le_ralentissement():
    intervalles = _calculer_intervalles(ROWS)
    assert len(intervalles) == 2
    premier, second = intervalles
    assert premier["t_debut"] == 0 and premier["t_fin"] == 24
    assert second["t_debut"] == 24 and second["t_fin"] == 48
    # la fermeture ralentit nettement entre le premier et le second intervalle
    assert second["vitesse_pct_h"] < premier["vitesse_pct_h"]


def test_format_intervalles_signale_le_plus_rapide_et_le_plus_lent():
    texte = _format_intervalles(_calculer_intervalles(ROWS))
    assert "0h-24h" in texte
    assert "24h-48h" in texte


def test_un_seul_point_ne_produit_aucun_intervalle():
    assert _calculer_intervalles(ROWS[:1]) == []
    assert "pas d'intervalle" in _format_intervalles([])


def test_phrase_heterogeneite_utilise_le_taux_global_pas_un_taux_d_intervalle():
    """Régression : le LLM a une fois recopié le taux d'un intervalle (1,58%/h) à la
    place du taux moyen global (0,83%/h) en composant cette phrase lui-même. La
    phrase est maintenant entièrement pré-calculée dans le code pour que ce genre
    d'erreur de recopie ne puisse plus se produire, quel que soit le LLM utilisé."""
    synthese = _calculer_synthese(ROWS)
    phrase = _phrase_heterogeneite(synthese, _calculer_intervalles(ROWS))
    assert "1,49%/h" in phrase  # taux moyen global, pas un taux d'intervalle
    assert "1,67%/h" in phrase  # intervalle le plus rapide (0h-24h)
    assert "1,32%/h" in phrase  # intervalle le plus lent (24h-48h)


def test_phrase_heterogeneite_absente_avec_un_seul_intervalle():
    synthese = _calculer_synthese(ROWS[:2])
    phrase = _phrase_heterogeneite(synthese, _calculer_intervalles(ROWS[:2]))
    assert phrase.startswith("(")


def test_format_intervalles_ne_produit_jamais_un_signe_double_sur_delta_negatif():
    """Régression : un delta négatif (réouverture) produisait littéralement
    "fermeture +-19,9 points de %" à cause d'un "+" codé en dur devant le nombre,
    injecté tel quel dans le prompt du LLM (corrigé le 2026-09-09)."""
    texte = _format_intervalles(_calculer_intervalles(ROWS_REOUVERTURE))
    assert "+-" not in texte


def test_format_intervalles_signale_la_reouverture_partielle():
    texte = _format_intervalles(_calculer_intervalles(ROWS_REOUVERTURE))
    assert "réouverture partielle" in texte
    assert "19,9" in texte


def test_phrase_reouverture_signale_le_bon_intervalle_et_la_bonne_valeur():
    """Retour biologiste (2026-09-09) : un delta négatif décrit en "taux plus
    faible" laisse croire à un simple ralentissement, alors que la surface non
    colonisée a réellement augmenté -- signalé explicitement en une phrase
    pré-calculée, comme _phrase_heterogeneite, plutôt que laissé au LLM."""
    phrase = _phrase_reouverture(_calculer_intervalles(ROWS_REOUVERTURE))
    assert "24h" in phrase and "48h" in phrase
    assert "19,9" in phrase


def test_phrase_reouverture_absente_sans_delta_negatif():
    phrase = _phrase_reouverture(_calculer_intervalles(ROWS))
    assert phrase.startswith("(")


def test_phrase_interpretation_conserve_le_signe_negatif_sur_reouverture():
    """Régression (2026-09-09) : même avec un patron explicite laissant le LLM
    remplir bornes/vitesses lui-même, mistral 7B a fabriqué un signe "-" ou inversé
    l'ordre des bornes. La phrase est maintenant entièrement pré-calculée, comme
    _phrase_heterogeneite, pour éliminer ce risque plutôt que le corriger par une
    consigne de prompt supplémentaire."""
    phrase = _phrase_interpretation(_calculer_intervalles(ROWS_REOUVERTURE))
    assert "0h-24h" in phrase and "0,9%/h" in phrase
    assert "24h-48h" in phrase and "-0,83%/h" in phrase


def test_phrase_interpretation_sans_reouverture_ne_fabrique_pas_de_signe():
    phrase = _phrase_interpretation(_calculer_intervalles(ROWS_SANS_REOUVERTURE))
    assert "24h-48h" in phrase and "0,83%/h" in phrase
    assert "-0,83" not in phrase
    assert "48h-24h" not in phrase


def test_phrase_interpretation_absente_avec_un_seul_intervalle():
    phrase = _phrase_interpretation(_calculer_intervalles(ROWS[:2]))
    assert phrase.startswith("(")


# Paragraphes réels observés en usage (2026-09-11) ou lors des isolation tests de
# PROMPT_CONTEXTE, dépassant la limite de "3 phrases" que le prompt seul n'a pas réussi à
# faire respecter de façon fiable (voir chatbot_prompt_fragility, mémoire projet).
PARAGRAPHE_5_PHRASES = (
    "Les articles considérés proposent des méthodes innovantes pour étudier les déplacements "
    "cellulaires lors de l'expérience de cicatrisation de plaie connue sous le nom de scratch "
    "assay. En utilisant des techniques d'imagerie quantitative et de suivi temporel, ils "
    "cherchent à caractériser la fermeture de la plaie et analyser la migration cellulaire. Ils "
    "proposent différentes approches, telles que la détection du bord de plaie par segmentation "
    "au niveau de contours et l'analyse de la migration en secteurs. Ces travaux montrent "
    "l'intérêt d'une approche quantitative et temporelle pour l'étude de la cicatrisation de "
    "plaie, mais les conditions biologiques, le système d'imagerie et les méthodes de "
    "quantification de ces travaux diffèrent de celles de notre expérience précise. Les "
    "résultats obtenus ici ne peuvent donc pas être directement comparés à ceux rapportés dans "
    "ces publications."
)

PARAGRAPHE_4_PHRASES = (
    "Les extraits fournis apportent un intérêt à l'imagerie quantitative et à l'analyse "
    "temporelle pour caractériser la fermeture de la plaie et la migration cellulaire dans le "
    "scratch assay. Ils proposent des méthodes originales pour extraire des trajectoires "
    "décrivant les dynamics de la fermeture de la plaie. En effet, ils utilisent des méthodes "
    "telles que le tracking de particules pseudo, la segmentation d'images à niveau de contour "
    "et le partage de l'image en secteurs pour décrire les migrations cellulaires dans le temps. "
    "Cependant, ces travaux étant effectués sous des conditions biologiques et technologiques "
    "différentes de l'expérience en cours, les résultats obtenus ne peuvent être directement "
    "comparés quantitativement."
)

PARAGRAPHE_3_PHRASES = (
    "Le travail présenté dans ces articles illustre l'intérêt d'utiliser des images "
    "quantitatives et une analyse temporelle pour caractériser la fermeture de la plaie et la "
    "migration cellulaire dans le scratch assay. Ils proposent des méthodes originales pour "
    "extraire des trajectoires de migration cellulaire et pour décrire la dynamique de la "
    "fermeture de la plaie. Toutefois, il est important de souligner que ces études ont été "
    "réalisées avec des systèmes d'imagerie, des méthodes de quantification et des conditions "
    "biologiques différentes de celles qui seront utilisées dans cette expérience, et donc que "
    "les résultats obtenus ici ne sont pas directement comparables à ceux rapportés dans ces "
    "publications."
)


def _compter_phrases(texte: str) -> int:
    import re
    return len([p for p in re.split(r"(?<=[.!?])\s+", texte.strip()) if p])


def test_limiter_paragraphe_contexte_reduit_5_phrases_a_3_en_gardant_la_cloture():
    resultat = limiter_paragraphe_contexte(PARAGRAPHE_5_PHRASES)
    assert _compter_phrases(resultat) == 3
    assert resultat.endswith(
        "Les résultats obtenus ici ne peuvent donc pas être directement comparés à ceux "
        "rapportés dans ces publications."
    )
    # La phrase intermédiaire (détail des méthodes) doit avoir disparu, pas les deux premières.
    assert "détection du bord de plaie par segmentation" not in resultat
    assert resultat.startswith("Les articles considérés proposent des méthodes innovantes")


def test_limiter_paragraphe_contexte_reduit_4_phrases_a_3_en_fusionnant_la_cloture():
    resultat = limiter_paragraphe_contexte(PARAGRAPHE_4_PHRASES)
    assert _compter_phrases(resultat) == 3
    assert "ne peuvent être directement comparés quantitativement" in resultat
    assert "tracking de particules pseudo" not in resultat


def test_limiter_paragraphe_contexte_ne_touche_pas_un_paragraphe_deja_conforme():
    assert limiter_paragraphe_contexte(PARAGRAPHE_3_PHRASES) == PARAGRAPHE_3_PHRASES


def test_limiter_paragraphe_contexte_tronque_sans_phrase_de_cloture_reconnaissable():
    texte = "Phrase un. Phrase deux. Phrase trois. Phrase quatre sans mot clé pertinent."
    resultat = limiter_paragraphe_contexte(texte)
    assert _compter_phrases(resultat) == 3
    assert resultat == "Phrase un. Phrase deux. Phrase trois."


# Variantes réelles du placeholder "(aucune réouverture observée...)" observées en usage
# (2026-09-11) fuitant dans "Interprétation prudente" -- voir chatbot_prompt_fragility, mémoire
# projet : deux corrections de PROMPT_RAPPORT ont été testées et rejetées (l'une n'a rien
# amélioré, l'autre a fait fabriquer une réouverture inexistante), la correction retenue est
# uniquement ce nettoyage après coup, sur le texte déjà généré.
def test_nettoyer_placeholder_reouverture_puce_isolee_entre_parentheses():
    texte = (
        "**Interprétation prudente**\n- Le taux moyen de fermeture ... cicatrisation.\n\n"
        "- (aucune réouverture observée sur les intervalles disponibles)\n\n"
        "Ce résumé décrit une évolution globale sur peu de mesures, pas une cinétique fine."
    )
    resultat = nettoyer_placeholder_reouverture(texte)
    assert "réouverture" not in resultat.lower()
    assert "Ce résumé décrit une évolution globale" in resultat
    assert "\n\n\n" not in resultat


def test_nettoyer_placeholder_reouverture_phrase_en_milieu_de_paragraphe():
    texte = (
        "Cette différence décrit uniquement les mesures disponibles et ne permet pas, à elle "
        "seule, de conclure à une accélération biologique de la cicatrisation. Aucune "
        "réouverture observée sur les intervalles disponibles. Ce résumé décrit une évolution "
        "globale sur peu de mesures, pas une cinétique fine."
    )
    resultat = nettoyer_placeholder_reouverture(texte)
    assert "réouverture" not in resultat.lower()
    assert "  " not in resultat
    assert resultat == (
        "Cette différence décrit uniquement les mesures disponibles et ne permet pas, à elle "
        "seule, de conclure à une accélération biologique de la cicatrisation. Ce résumé décrit "
        "une évolution globale sur peu de mesures, pas une cinétique fine."
    )


def test_nettoyer_placeholder_reouverture_ne_touche_pas_une_vraie_reouverture():
    texte = (
        "Réouverture partielle détectée sur les mesures disponibles : la surface non colonisée "
        "a augmenté de 19,9 points de % entre 24h et 48h. Cela peut correspondre à un vrai "
        "phénomène biologique."
    )
    assert nettoyer_placeholder_reouverture(texte) == texte


def test_nettoyer_placeholder_reouverture_inchange_sans_placeholder():
    texte = "Un résumé quelconque sans aucun rapport avec une réouverture de plaie."
    assert nettoyer_placeholder_reouverture(texte) == texte


def test_corriger_accord_conditions_biologiques_pluralise_celle_utilisee():
    texte = (
        "les conditions biologiques de ces travaux diffèrent de celle utilisée dans "
        "cette expérience, et que les résultats obtenus ici ne sont donc pas comparables."
    )
    resultat = corriger_accord_conditions_biologiques(texte)
    assert "celles utilisées" in resultat
    assert "celle utilisée" not in resultat


def test_corriger_accord_conditions_biologiques_ne_touche_pas_un_celle_sans_rapport():
    texte = "Celle utilisée dans cette expérience est différente de la précédente."
    assert corriger_accord_conditions_biologiques(texte) == texte


def test_detecter_contradiction_reouverture_signale_augmentation_sans_vraie_reouverture():
    # Cas réel (2026-09-12) : fermeture 0% -> 65,7%, donc la surface non colonisée
    # diminue tout du long -- affirmer qu'elle "a augmenté" est une contradiction.
    intervalles = _calculer_intervalles(ROWS)
    texte = "La surface non colonisée a augmenté au fil du temps, avec des variations."
    assert detecter_contradiction_reouverture(texte, intervalles) is True


def test_detecter_contradiction_reouverture_absente_texte_correct():
    intervalles = _calculer_intervalles(ROWS)
    texte = "La surface non colonisée a diminué au fil du temps, avec une différence entre les taux."
    assert detecter_contradiction_reouverture(texte, intervalles) is False


def test_detecter_contradiction_reouverture_ne_signale_pas_une_vraie_reouverture():
    # ROWS_REOUVERTURE contient un vrai delta négatif : "a augmenté" y est légitime
    # (c'est la formulation même de _phrase_reouverture), ne doit jamais être signalé.
    intervalles = _calculer_intervalles(ROWS_REOUVERTURE)
    texte = "Réouverture partielle détectée : la surface non colonisée a augmenté entre 24h et 48h."
    assert detecter_contradiction_reouverture(texte, intervalles) is False


def test_detecter_contradiction_reouverture_absente_sans_mention_de_surface():
    intervalles = _calculer_intervalles(ROWS)
    texte = "Cette différence décrit uniquement les mesures disponibles, sans conclure à une accélération."
    assert detecter_contradiction_reouverture(texte, intervalles) is False
