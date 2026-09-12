"""Assistant IA : répond aux questions du biologiste en confrontant le résultat
de segmentation obtenu à la littérature scientifique sur la cicatrisation.

Utilise un LLM local via Ollama (modèle mistral) pour ne dépendre d'aucune clé
d'API externe, et index_articles.chercher() pour le contexte scientifique
(retrieval-augmented generation).

Usage:
    python chatbot/chatbot.py
"""
import queue
import re
import sys
import threading
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).parent))
from index_articles import chercher

MODEL = "mistral"

# Délai max entre deux morceaux du flux avant d'abandonner (pas un délai sur la
# durée totale) : observé en pratique, Ollama peut se bloquer en cours de
# génération sans jamais renvoyer d'erreur ni de nouvel octet (probablement une
# contention GPU quand une segmentation d'image PyTorch vient de tourner juste
# avant dans la même session). Le timeout intégré au client HTTP d'ollama-python
# ne suffit pas à détecter ça (testé : blocage de 180s+ sans qu'il se déclenche
# pour un appel en streaming), donc la génération tourne dans un thread séparé
# et ce module surveille lui-même, via une file, qu'un nouveau morceau arrive
# bien avant CHUNK_TIMEOUT_S secondes.
CHUNK_TIMEOUT_S = 45.0
_client = ollama.Client()

# Règle globale ajoutée le 2026-09-12 (retour utilisateur, capture d'écran du chat interactif) :
# une première tentative de correction avait ciblé uniquement la section "Littérature" (qui
# interdisait déjà "conforme à la littérature"/"en accord"/"confirme"), mais un test d'isolation
# (3 générations baseline vs 3 candidat) a montré que le LLM place ces généralisations/fabrications
# dans "Interprétation" à la place -- section qui n'avait jamais cette interdiction. Pire : le
# candidat v1 laissait passer une fabrication plus grave qu'à l'origine ("un taux de fermeture de
# 80% à 90% après 72 heures est souvent rapporté", un chiffre inventé absent des 3 extraits réels).
# La règle a donc été déplacée dans "Consignes valables dans les 4 sections" (s'applique aussi à
# Interprétation) et étendue pour interdire explicitement d'inventer un chiffre de comparaison.
# Retesté (3 vs 3, même question de la capture) : baseline 3/3 générations avec un problème (dont
# une plage de valeurs entièrement fabriquée, "20% à 100% en 7 à 21 jours"), candidat 1/3 seulement
# (et ce cas résiduel s'auto-corrige immédiatement dans la même phrase : "cependant, il est
# difficile de comparer directement..."). Amélioration nette, jamais une élimination totale --
# cohérent avec toutes les autres itérations de prompt de ce projet. Gardé.
PROMPT_TEMPLATE = """Tu es un assistant scientifique qui aide un biologiste à interpréter un \
résultat de segmentation automatique de plaie (wound healing assay), en répondant à sa question.

Résultat de segmentation obtenu par le biologiste :
{resultat_segmentation}

Question du biologiste :
{question}

Extraits d'articles scientifiques pertinents :
{contexte}

Réponds en français simple, clair, pour un biologiste non spécialiste en IA. Si le message est \
une vraie question scientifique, structure TOUJOURS ta réponse en EXACTEMENT 4 sections, avec ces \
4 titres en gras et dans cet ordre (rien avant, rien après, aucune section fusionnée ni omise) — \
chaque section ne doit contenir QUE le type d'information indiqué, pour que le biologiste \
distingue toujours ce qui est mesuré, ce qui en est déduit, ce qui reste incertain, et ce qui \
vient de la littérature :

**📊 Résultats observés**
- S'il y a un résultat de segmentation ci-dessus, reprends UNIQUEMENT les valeurs mesurées
  pertinentes pour la question, telles quelles (ne les recalcule pas, ne les arrondis pas
  différemment). Aucune interprétation ici.
- S'il n'y a pas de résultat ci-dessus (question générale, aucune image analysée), écris
  "Aucune image analysée pour cette question." et rien d'autre dans cette section.

**🔬 Interprétation**
- Réponds directement à la question du biologiste, en t'appuyant sur les résultats ci-dessus
  (si présents) et sur les extraits scientifiques (mécanismes, méthodes, ordres de grandeur
  généraux). Reste prudent : décris ce que les données/la littérature suggèrent, pas une
  certitude.

**⚠️ Limites**
- Dis explicitement ce que la question posée et les données disponibles ne permettent PAS de
  conclure (ex : nombre de mesures insuffisant pour une cinétique fine, absence de réplicats,
  extraits trop généraux pour une comparaison chiffrée directe). Une phrase suffit si les
  limites sont déjà couvertes ailleurs dans la réponse.

**📚 Littérature**
- Cite toujours le titre complet de l'article entre guillemets à chaque mention. Ne dis jamais
  "le premier article" ou "le deuxième article" : ces numéros ne correspondent à rien pour le
  lecteur et prêtent à confusion. N'invente jamais de lien ni d'URL (les extraits n'en
  fournissent pas).
- N'affirme JAMAIS qu'un résultat est "conforme à la littérature", "en accord avec les études",
  un "résultat normal", ou qu'il "confirme" quoi que ce soit scientifiquement, sauf si un extrait
  décrit explicitement des conditions expérimentales comparables (même type de test, échelle de
  temps comparable) ET une valeur chiffrée directement comparable. C'est le cas le plus rare : les
  extraits sont en général des résumés généraux (mécanismes, méthodes), pas des points de
  comparaison chiffrés pour les conditions précises du biologiste. Par défaut, dis explicitement
  que les extraits fournissent un contexte général mais ne permettent pas de conclure à une
  concordance quantitative, plutôt que d'inventer un rapprochement.
- Si aucun extrait ci-dessus n'apporte d'éclairage réellement pertinent pour cette question
  précise, écris-le explicitement (ex : "le corpus documentaire disponible ne permet pas de
  mettre ce point en contexte de façon pertinente ici") plutôt que de forcer un lien approximatif.

Consignes valables dans les 4 sections :
- N'affirme JAMAIS, dans AUCUNE des 4 sections (y compris Interprétation), qu'une valeur est
  "conforme à la littérature", "en accord avec les études", une "moyenne normale", ou qu'elle
  est "souvent"/"généralement"/"habituellement"/"typiquement" observée/rapportée dans la
  littérature -- sauf si un extrait ci-dessus le dit EXPLICITEMENT avec sa propre valeur
  chiffrée citée. N'invente JAMAIS un chiffre de comparaison (pourcentage, durée, vitesse...)
  absent des extraits fournis, même à titre d'exemple ou d'ordre de grandeur.
- Ne pose jamais de question de clarification en retour (pas de "pouvez-vous préciser...",
  "quelle est la durée de l'expérience ?", etc.) : réponds directement avec le résultat de
  segmentation et les extraits déjà fournis ci-dessus. S'il manque une information pour répondre
  complètement, dis-le en une phrase (section Limites) et réponds quand même du mieux possible
  avec ce qui est disponible, au lieu de renvoyer la question au biologiste.
- Reste factuel et concis (une à trois phrases par section).

Exception à la structure en 4 sections : si le message n'est pas une vraie question scientifique
(simple salutation comme "bonjour", message vide de sens, remerciement...), ignore tout ce qui
précède et réponds brièvement et simplement (ex : salue en retour, invite à poser une question
sur la cicatrisation), sans sections ni lien forcé avec les extraits.
"""


def _format_resultat(resultat_segmentation: dict | None) -> str:
    if not resultat_segmentation:
        return "(aucun : question générale, sans image analysée)"
    return "\n".join(f"- {cle} : {valeur}" for cle, valeur in resultat_segmentation.items())


def _format_contexte(articles: list[dict]) -> str:
    # Pas de crochets autour du titre : un modèle comme mistral a tendance à
    # recopier ce gabarit tel quel comme "citation", au lieu de suivre la
    # consigne de citer le titre entre guillemets dans sa réponse.
    return "\n\n".join(
        f"Titre : {a['titre']}\nAnnée : {a['annee']}\nExtrait : {a['extrait']}"
        for a in articles
    )


def _intro_articles(articles: list[dict]) -> str:
    """Ligne fixe listant les titres réellement retournés par le retrieval, entre
    guillemets, composée en Python plutôt que laissée à la charge du LLM. Constaté
    en pratique (retour biologiste, 2026-09-09) : même avec une consigne "OBLIGATOIRE"
    de citer au moins un titre entre guillemets, mistral 7B omet parfois toute
    citation (ex: "un article de 2023" sans titre ni guillemets) ou recopie un titre
    tronqué sans guillemets -- pas un risque de chiffre fabriqué, mais un défaut de
    forme qu'une consigne de prompt supplémentaire ne suffisait pas à éliminer de
    façon fiable. Affichée avant le paragraphe de synthèse du LLM, qui n'a donc plus
    à se charger lui-même de la citation."""
    if not articles:
        return "(aucun article disponible)"
    titres = [f"« {a['titre']} » ({a['annee']})" for a in articles]
    if len(titres) == 1:
        return f"Article considéré : {titres[0]}."
    return "Articles considérés : " + ", ".join(titres[:-1]) + " et " + titres[-1] + "."


_ARRET = object()  # sentinelle : signale la fin normale du flux dans la file


BATCH_INTERVAL_S = 0.5  # regroupe les morceaux reçus sur cette fenêtre avant de les afficher


def _stream_ollama(prompt: str):
    """Lance une génération Ollama en streaming ; lève RuntimeError immédiatement
    si Ollama est injoignable (avant de renvoyer le générateur de morceaux).

    La génération tourne dans un thread à part qui pousse chaque morceau dans une
    file ; ce générateur lit la file avec un timeout et regroupe les morceaux reçus
    sur BATCH_INTERVAL_S avant de les céder. Le regroupement est nécessaire : testé
    en conditions réelles, céder un morceau à chaque token (observé : ~260 morceaux
    sur ~60s) fait décrocher l'affichage st.write_stream côté navigateur en cours de
    route (Streamlit cesse de transmettre les mises à jour) alors que la génération
    elle-même se termine normalement côté serveur — un thread de debug l'a confirmé.

    Si Ollama se bloque en cours de route (aucun nouveau morceau pendant plus de
    CHUNK_TIMEOUT_S), on abandonne proprement avec un message d'avertissement ajouté
    au texte déjà reçu, plutôt que de laisser l'appli figée indéfiniment. Le thread
    bloqué est laissé mourir en arrière-plan (démon) : on n'attend pas qu'il se
    débloque."""
    try:
        stream = _client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], stream=True)
    except Exception as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama. Vérifiez qu'Ollama est installé et lancé "
            f"(ollama serve) et que le modèle est disponible (ollama pull {MODEL})."
        ) from exc

    file_morceaux = queue.Queue()

    def producteur():
        try:
            for chunk in stream:
                file_morceaux.put(chunk["message"]["content"])
        except Exception:
            pass
        finally:
            file_morceaux.put(_ARRET)

    threading.Thread(target=producteur, daemon=True).start()

    def morceaux():
        # Sondage court (POLL_S) pour pouvoir vérifier régulièrement à la fois le
        # délai depuis le dernier morceau reçu (détection de blocage, CHUNK_TIMEOUT_S)
        # et le délai depuis le dernier envoi au navigateur (regroupement d'affichage,
        # BATCH_INTERVAL_S) sans dépendre d'un seul timeout pour les deux.
        POLL_S = 0.1
        tampon = []
        dernier_item = time.monotonic()
        dernier_envoi = time.monotonic()

        while True:
            try:
                item = file_morceaux.get(timeout=POLL_S)
            except queue.Empty:
                item = None

            maintenant = time.monotonic()

            if item is not None:
                if item is _ARRET:
                    if tampon:
                        yield "".join(tampon)
                    return
                tampon.append(item)
                dernier_item = maintenant

            if maintenant - dernier_item > CHUNK_TIMEOUT_S:
                if tampon:
                    yield "".join(tampon)
                yield (
                    "\n\n⚠️ *Génération interrompue : le modèle local n'a pas répondu à temps "
                    "(Ollama a peut-être été surchargé, par exemple juste après une analyse "
                    "d'image). Réessayez.*"
                )
                return

            if tampon and maintenant - dernier_envoi >= BATCH_INTERVAL_S:
                yield "".join(tampon)
                tampon = []
                dernier_envoi = maintenant

    return morceaux()


# Ordinaux français utilisés pour désigner un article par sa position plutôt que par son
# titre ("le premier article", "le second article"...) -- interdit depuis l'origine du
# prompt (PROMPT_CONTEXTE) mais contourné par le LLM de temps à autre en usage réel
# (2026-09-10). Couvre aussi "la première étude"/"la seconde étude" (synonyme d'"article"
# employé par mistral, jamais couvert jusqu'ici -- trouvé par l'isolation test du
# PROMPT_CONTEXTE reformulé le 2026-09-11, voir chatbot_prompt_fragility en mémoire projet ;
# le trou existait déjà avec l'ancien prompt, simplement jamais exercé par les isolation
# tests précédents). Index = position dans `sources` (même ordre que dans _intro_articles /
# _format_contexte).
_ORDINAUX_ARTICLE = {
    "premier": 0, "premiere": 0, "première": 0,
    "deuxieme": 1, "deuxième": 1, "second": 1, "seconde": 1,
    "troisieme": 2, "troisième": 2,
    "quatrieme": 3, "quatrième": 3,
}
_REF_ORDINALE_RE = re.compile(
    r"\b(?:le|la)\s+(premi[eè]re?|deuxi[eè]me|second[e]?|troisi[eè]me|quatri[eè]me)\s+(?:articles?|[ée]tudes?)\b",
    re.IGNORECASE,
)


def corriger_references_ordinales(texte: str, sources: list[str]) -> str:
    """Remplace après coup "le premier/second/troisième article" (ou "étude", synonyme
    parfois employé à la place) par son vrai titre entre guillemets, dans l'ordre où les
    articles ont été présentés au LLM (même ordre que `sources`).

    Constaté en usage réel (2026-09-10, retour biologiste) : PROMPT_CONTEXTE
    interdit explicitement de citer un article par sa position depuis son origine,
    mais mistral 7B contourne parfois cette règle malgré tout. Renforcer encore la
    consigne risquerait de déstabiliser d'autres sections (voir
    chatbot_prompt_fragility en mémoire projet -- le prompt est déjà considéré à
    son plafond de complexité sûre) -- corrigé après coup en code pur à la place,
    sur le même principe que detecter_citations_suspectes ci-dessous : aucune
    modification du prompt ni du contexte, donc aucun risque de régression sur les
    sections numériques.

    Si l'index déduit dépasse le nombre de sources réelles (ex: "le quatrième
    article" alors qu'il n'y en a que 3), la référence est laissée telle quelle
    plutôt que de risquer un remplacement incorrect.
    """
    def _remplacement(m: re.Match) -> str:
        idx = _ORDINAUX_ARTICLE.get(m.group(1).lower())
        if idx is None or idx >= len(sources):
            return m.group(0)
        remplacement = f"l'article « {sources[idx]} »"
        if m.group(0)[0].isupper():
            remplacement = remplacement[0].upper() + remplacement[1:]
        return remplacement

    return _REF_ORDINALE_RE.sub(_remplacement, texte)


# Accord grammatical observé en usage réel (2026-09-12, retour utilisateur) dans la phrase de
# clôture du paragraphe "Mise en contexte scientifique" : PROMPT_CONTEXTE demande de citer
# ensemble "le protocole, le système d'imagerie, les méthodes de quantification et les
# conditions biologiques" (liste au pluriel), mais mistral 7B paraphrase parfois cette liste
# avec un pronom/participe au singulier ("...diffèrent de celle utilisée dans cette expérience"
# au lieu de "...diffèrent de celles utilisées..."). Corrigé après coup en code pur, sur le même
# principe que corriger_references_ordinales ci-dessus : ne touche ni au prompt ni au contexte
# retrouvé (juste un accord grammatical sur du texte déjà généré), donc aucun test d'isolation
# nécessaire.
_ACCORD_CELLE_RE = re.compile(
    r"\b(diff[eè]r(?:e|ent))\s+de\s+celle\s+utilis[ée]e?s?\b", re.IGNORECASE
)


def corriger_accord_conditions_biologiques(texte: str) -> str:
    """Remplace "diffèrent de celle utilisée" par "diffèrent de celles utilisées"
    (accord pluriel), quand ce pronom fait référence à la liste au pluriel du
    protocole/système d'imagerie/méthodes/conditions biologiques dans la phrase de
    clôture du paragraphe "Mise en contexte scientifique". Portée volontairement
    restreinte au tour "diffère(nt) de celle..." (jamais "celle utilisée" seule
    ailleurs dans le texte) pour ne jamais corriger un usage légitime et sans
    rapport de "celle" au singulier."""
    def _remplacement(m: re.Match) -> str:
        return f"{m.group(1)} de celles utilisées"

    return _ACCORD_CELLE_RE.sub(_remplacement, texte)


# Correction après coup de la longueur du paragraphe "Mise en contexte scientifique" :
# PROMPT_CONTEXTE exige "EXACTEMENT 3 phrases", mais mistral 7B dépasse cette limite dans
# environ 1/3 des générations (jusqu'à 5 phrases observées en usage réel, 2026-09-11). Deux
# reformulations du prompt ont été testées en isolation pour éliminer ce dépassement -- l'une
# n'a rien amélioré, l'autre a réintroduit une généralisation biologique non garantie
# ("comportement coordonné des cellules") pire que le problème visé (voir
# chatbot_prompt_fragility, mémoire projet). Corrigé à la place en code pur, sur le même
# principe que corriger_references_ordinales ci-dessus : jamais de fabrication à corriger ici
# (juste une longueur), donc aucun risque de régression sur les sections numériques.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# "compar" n'apparaît légitimement que dans la phrase de clôture obligatoire du prompt ("ne
# sont pas directement comparables..."), puisque le prompt interdit explicitement toute
# comparaison ailleurs dans le paragraphe -- un simple test de sous-chaîne suffit donc à la
# repérer de façon fiable, sans dépendre d'une formulation exacte.
_MOT_COMPARAISON_RE = re.compile(r"compar", re.IGNORECASE)
_MOT_DIFFERENCE_RE = re.compile(
    r"(protocole|conditions? biologiques?|syst[eè]me[s]? d'imagerie|m[eé]thodes? de quantification).{0,60}diff[ée]r",
    re.IGNORECASE,
)


def limiter_paragraphe_contexte(texte: str, max_intro: int = 2) -> str:
    """Réduit le paragraphe "Mise en contexte scientifique" à 3 phrases au maximum,
    en conservant TOUJOURS la phrase de clôture (non-comparabilité) plutôt que de
    tronquer aveuglément aux 3 premières phrases -- une simple troncature couperait
    la phrase la plus importante du paragraphe (celle qui prévient toute
    surinterprétation) si elle arrive en 4e ou 5e position.

    Ne modifie rien si le paragraphe fait déjà 3 phrases ou moins, ou si aucune
    phrase de clôture reconnaissable n'est trouvée (dans ce cas, tronque aux 3
    premières phrases plutôt que de risquer de perdre la conclusion)."""
    phrases = [p for p in _SENTENCE_SPLIT_RE.split(texte.strip()) if p]
    if len(phrases) <= 3:
        return texte

    idx_cloture = next((i for i, p in enumerate(phrases) if _MOT_COMPARAISON_RE.search(p)), None)
    if idx_cloture is None:
        return " ".join(phrases[:3])

    # La phrase de clôture est parfois précédée d'une phrase séparée qui énonce la
    # différence de protocole/conditions avant la conclusion "donc non comparable" :
    # les deux sont alors gardées ensemble plutôt que de couper cette liaison.
    if idx_cloture > 0 and _MOT_DIFFERENCE_RE.search(phrases[idx_cloture - 1]):
        idx_cloture -= 1

    intro = phrases[:min(max_intro, idx_cloture)]
    cloture = phrases[idx_cloture:]
    return " ".join(intro + cloture)


# Repère les citations entre guillemets (français « » ou droits/courbes "..."/"...")
# assez longues pour être un titre d'article plutôt qu'un mot mis en emphase.
_CITATION_RE = re.compile(r"[«\"“]([^»\"”]{15,300})[»\"”]")
SEUIL_SIMILARITE_CITATION = 0.7  # ratio SequenceMatcher au-dessus duquel deux titres sont "le même"


def _normaliser_titre(texte: str) -> str:
    """Normalise un titre pour la comparaison : enlève les accents/diacritiques
    (variantes unicode entre le titre stocké et sa recopie par le LLM), casse et
    espaces superflus."""
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return " ".join(texte.lower().split())


def detecter_citations_suspectes(texte: str, sources: list[str]) -> list[str]:
    """Repère, dans un texte déjà généré, les citations entre guillemets qui ne
    correspondent à aucun des titres réellement retournés par le retrieval
    (`sources`) -- signe qu'un titre a été inventé par le LLM plutôt que recopié
    depuis le contexte fourni (observé en usage réel : un titre plausible mais
    absent du corpus, voir chatbot_prompt_fragility en mémoire projet).

    Vérification purement programmatique APRÈS génération : ne modifie ni le
    prompt ni la génération elle-même, donc n'ajoute aucun des risques déjà
    documentés (toute modification du contexte/prompt doit être validée par un
    test d'isolation ; une simple relecture du texte produit n'a pas cette
    contrainte).

    Retourne la liste des citations suspectes trouvées dans `texte` (vide si
    aucune, y compris si `texte` ne contient aucune citation entre guillemets).
    """
    sources_norm = [_normaliser_titre(s) for s in sources]
    suspectes = []
    for citation in _CITATION_RE.findall(texte):
        citation_norm = _normaliser_titre(citation)
        correspond_a_une_source = any(
            citation_norm in source_norm
            or source_norm in citation_norm
            or SequenceMatcher(None, citation_norm, source_norm).ratio() >= SEUIL_SIMILARITE_CITATION
            for source_norm in sources_norm
        )
        if not correspond_a_une_source:
            suspectes.append(citation.strip())
    return suspectes


# Repère une affirmation d'augmentation de la surface non colonisée / de la plaie, quel que
# soit le nom exact utilisé par le LLM pour la désigner ("la surface non colonisée",
# "la surface de la plaie", "la zone non colonisée"...). Attention : cette même formulation
# ("a augmenté") est aussi celle, légitime, utilisée par _phrase_reouverture quand une vraie
# réouverture est détectée -- la détection ci-dessous ne doit donc jamais se fier au seul mot,
# voir detecter_contradiction_reouverture.
_AUGMENTATION_SURFACE_RE = re.compile(
    r"\b(?:surface|zone)[^.]{0,40}\ba\s+augment[ée]e?\b",
    re.IGNORECASE,
)


def detecter_contradiction_reouverture(texte: str, intervalles: list[dict]) -> bool:
    """Détecte une contradiction factuelle observée en usage réel (2026-09-12, retour
    utilisateur) : le LLM affirme dans "Interprétation prudente" que "la surface non
    colonisée a augmenté au fil du temps", alors qu'aucun intervalle calculé ne montre
    en réalité de delta de fermeture négatif -- la fermeture est passée de 0 % à 65,7 %
    sur l'exemple observé, donc la surface non colonisée DIMINUE, pas l'inverse.

    Ne peut pas se contenter de chercher le mot "augmenté" : c'est aussi la formulation
    légitime de _phrase_reouverture quand une vraie réouverture existe (delta négatif sur
    un intervalle réel). Cette fonction ne signale donc que le cas contradictoire : le texte
    affirme une augmentation alors qu'aucun intervalle réel n'en montre une.

    Vérification purement programmatique après génération (comme detecter_citations_suspectes
    ci-dessus) : ne modifie ni le prompt ni le texte, se contente de signaler."""
    reouverture_reelle = any(iv["delta_closure_pct"] < 0 for iv in intervalles)
    if reouverture_reelle:
        return False
    return bool(_AUGMENTATION_SURFACE_RE.search(texte))


LONGUEUR_MIN_QUESTION = 8  # en dessous, ce n'est pas une vraie question (salutation, faute de frappe...)

REPONSE_SALUTATION = (
    "Bonjour ! Posez-moi une question sur la cicatrisation ou sur un résultat de "
    "segmentation (surface, vitesse de fermeture...) et je chercherai la littérature "
    "scientifique pertinente pour vous répondre."
)


def _question_triviale(question: str) -> bool:
    """Un message trop court pour être une vraie question (ex: "bnj", "ok") ne doit
    pas déclencher toute la recherche RAG + génération : un petit modèle local comme
    mistral 7B ne suit pas de façon fiable la consigne "ne force pas de réponse
    scientifique à une salutation", donc ce filtre est fait en code plutôt qu'en
    comptant sur le LLM."""
    return len(question.strip()) < LONGUEUR_MIN_QUESTION


def _construire_prompt(resultat_segmentation: dict | None, question: str, n_articles: int):
    # hybride=False (décision finale, 2026-09-08) : activé un temps sur demande
    # explicite malgré un risque mesuré (voir chatbot_prompt_fragility en mémoire
    # projet), puis revenu en arrière après reconsidération -- pour un usage
    # scientifique sérieux, le risque de chiffres fabriqués l'emporte sur le gain
    # de retrieval. Le mode hybride reste disponible et validé pour le retrieval
    # seul (voir evaluate_retrieval.py), pas pour la génération.
    articles = chercher(question, n=n_articles, hybride=False)
    prompt = PROMPT_TEMPLATE.format(
        resultat_segmentation=_format_resultat(resultat_segmentation),
        question=question,
        contexte=_format_contexte(articles),
    )
    return prompt, articles


def repondre(resultat_segmentation: dict | None, question: str, n_articles: int = 3) -> dict:
    """Génère une réponse à la question du biologiste, appuyée sur la littérature.

    resultat_segmentation: dict des métriques déjà calculées par l'app
    (ex: {"surface_px2": 12450, "fermeture_pct": 42.0, "duree_h": 48}), ou None
    pour une question générale sans image analysée.

    Retourne {"reponse": str, "sources": [titres d'articles utilisés]}.
    """
    if _question_triviale(question):
        return {"reponse": REPONSE_SALUTATION, "sources": []}

    prompt, articles = _construire_prompt(resultat_segmentation, question, n_articles)

    try:
        response = _client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    except Exception as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama. Vérifiez qu'Ollama est installé et lancé "
            f"(ollama serve) et que le modèle est disponible (ollama pull {MODEL})."
        ) from exc

    return {
        "reponse": response["message"]["content"],
        "sources": [a["titre"] for a in articles],
    }


def repondre_stream(resultat_segmentation: dict | None, question: str, n_articles: int = 3):
    """Comme repondre(), mais renvoie la réponse morceau par morceau au lieu
    d'attendre le texte complet : une réponse mistral en local prend 60-90s,
    et un écran figé pendant tout ce temps donne l'impression que l'appli est
    plantée. Affichée en direct (ex: st.write_stream côté app.py), la réponse
    apparaît mot à mot dès les premières secondes, comme dans ChatGPT.

    Retourne (générateur de morceaux de texte, sources). Lève RuntimeError
    immédiatement si Ollama est injoignable (avant de renvoyer le générateur).
    """
    if _question_triviale(question):
        return iter([REPONSE_SALUTATION]), []

    prompt, articles = _construire_prompt(resultat_segmentation, question, n_articles)
    return _stream_ollama(prompt), [a["titre"] for a in articles]


PROMPT_RAPPORT = """Tu es un assistant scientifique qui aide un biologiste à rédiger un premier \
résumé de son expérience de cicatrisation de plaie (wound healing / scratch assay), à partir de \
mesures de surface obtenues automatiquement par segmentation d'image sur une série d'images prises \
à différents temps.

Synthèse chiffrée déjà calculée par le pipeline (valeurs finales à reprendre telles quelles dans \
ta réponse ; ne recalcule et n'invente aucun de ces chiffres, ne les arrondis pas différemment) :
{synthese}

Vitesse de fermeture par intervalle, déjà calculée par le pipeline (sers-t'en pour comparer les \
vitesses moyennes entre intervalles, sans recalculer ces vitesses toi-même) :
{intervalles}

Phrase de liaison entre la vitesse moyenne globale et l'hétérogénéité par intervalle, déjà \
composée par le pipeline (à reprendre mot pour mot dans "Analyse quantitative" si elle contient \
une vraie phrase ; si elle est entre parenthèses, ne la reprends pas et n'en invente pas \
d'équivalent) :
{phrase_heterogeneite}

Détail des mesures par point de temps (pour décrire la tendance point par point uniquement) :
{mesures}

Phrase de comparaison des intervalles, déjà composée par le pipeline (à reprendre mot pour mot \
comme première phrase de "Interprétation prudente" si elle contient une vraie phrase ; si elle est \
entre parenthèses, n'invente pas de comparaison à la place) :
{phrase_interpretation}

Réouverture partielle éventuelle, déjà détectée par le pipeline (à reprendre mot pour mot dans \
"Interprétation prudente" si elle contient une vraie phrase ; si elle est entre parenthèses, \
n'invente pas de réouverture à la place) :
{phrase_reouverture}

Rédige un résumé scientifique structuré EN EXACTEMENT 3 sections, avec ces 3 titres en gras et \
dans cet ordre (rien avant, rien après, aucune section fusionnée ni omise) — chaque section ne \
doit contenir QUE le type d'information indiqué, pour que le biologiste distingue toujours ce qui \
est mesuré, ce qui est calculé, et ce qui reste prudent :

**Résultats observés**
- 1 à 2 phrases, UNIQUEMENT les valeurs mesurées telles quelles : nombre de mesures, surface
  initiale et finale, fermeture finale en %, durée totale d'observation. Reprends-les exactement
  depuis la synthèse chiffrée ci-dessus (ligne "Nombre de mesures" incluse), sans les recalculer
  ni en changer aucune. Aucune interprétation ici.
- n'affirme jamais que "le biologiste a effectué l'expérience" ou toute variante supposant qui a
  réalisé la manipulation (l'application ne le sait pas) : décris les mesures elles-mêmes, par
  exemple "les mesures analysées proviennent d'un essai de type scratch assay" ou "l'analyse des
  images issues d'un scratch assay montre une diminution de la surface non colonisée".

**Analyse quantitative**
- Cite TOUJOURS la vitesse moyenne de fermeture sur l'ensemble de la période (depuis la synthèse
  chiffrée), même si tu détailles ensuite les taux par intervalle : ne la laisse jamais implicite
  ou absente. Précise aussi quel intervalle a le taux le plus élevé / le plus faible (depuis les
  vitesses par intervalle ci-dessus), sans te contenter de répéter les chiffres déjà visibles dans
  le tableau du biologiste.
- Si la "Phrase de liaison" ci-dessus contient une vraie phrase (pas une note entre parenthèses),
  REPRENDS-LA MOT POUR MOT dans cette section, sans changer un seul chiffre et sans la recomposer
  toi-même : ne recopie jamais le taux d'un intervalle à la place de la vitesse moyenne globale dans
  cette phrase, cette phrase fait seule autorité sur ce point précis. Sans elle, un lecteur qui compare le
  résumé au tableau des mesures peut croire à un oubli plutôt qu'à un choix de présentation. Si elle
  est entre parenthèses, n'invente pas de comparaison d'hétérogénéité à la place.
- 2 à 3 phrases au total pour cette section.
- IMPORTANT : chaque vitesse en %/h que tu écris doit être copiée mot pour mot depuis la section
  "Vitesse de fermeture par intervalle" ci-dessus (ou depuis la synthèse chiffrée pour la vitesse
  moyenne) : ce sont les deux seuls endroits de ce prompt qui font autorité sur les vitesses.
  N'écris jamais une vitesse qui ne soit pas recopiée telle quelle depuis ces deux sections, SIGNE
  INCLUS : recopie le signe "-" si la source en affiche un (une vitesse négative signifie que la
  surface a augmenté, pas seulement fermé plus lentement), et n'en ajoute JAMAIS un si la source
  n'en affiche pas -- dans les deux cas, le signe de ta phrase doit être identique à celui affiché
  dans la source, jamais déduit ni copié depuis un autre exemple.
- Reste un calcul, pas une interprétation biologique : ne parle pas encore ici d'accélération,
  de ralentissement, ni de ce que ça signifie pour la cicatrisation (voir section suivante).

**Interprétation prudente**
- S'il y a une "Phrase de comparaison des intervalles" fournie plus bas (pas une note entre
  parenthèses), REPRENDS-LA MOT POUR MOT comme première phrase de cette section, SANS CHANGER NI
  RECALCULER NI RECOMPOSER un seul mot, un seul chiffre, un seul signe ou l'ordre des bornes : cette
  phrase fait seule autorité, ne la remplace jamais par ta propre comparaison des intervalles même
  si tu penses pouvoir la reformuler plus clairement. Si elle est entre parenthèses, n'invente aucune
  comparaison à la place. N'écris JAMAIS les mots "accélération", "accéléré", "ralentissement" ou
  "ralenti" au sujet de la cicatrisation elle-même ailleurs dans cette section, et n'écris jamais que
  "la plaie cicatrise plus vite/plus lentement",
- Si la "Réouverture partielle" ci-dessus contient une vraie phrase (pas une note entre
  parenthèses), REPRENDS-LA MOT POUR MOT, EN PLUS du reste de cette section, sans changer un seul
  chiffre et sans la recomposer toi-même : c'est une information factuelle importante pour le
  biologiste (la surface non colonisée a augmenté, pas seulement "ralenti"), jamais à passer sous
  silence ni à reformuler en "ralentissement". Si elle est entre parenthèses, n'invente aucune
  réouverture à la place.
- rappelle en une phrase, SANS RÉPÉTER LE CHIFFRE (déjà donné dans "Résultats observés" ci-dessus,
  ne le recompte ni ne l'invente pas une seconde fois ici), que ce résumé décrit une évolution
  globale sur peu de mesures, pas une cinétique fine,
- n'affirme jamais une tendance qui contredirait les vitesses par intervalle fournies ci-dessus.

Reste factuel et concis (4 à 7 phrases au total sur les 3 sections), sans réclamer d'information
supplémentaire au biologiste.

RÈGLE GLOBALE VALABLE DANS LES 3 SECTIONS CI-DESSUS, PAS SEULEMENT DANS "Interprétation prudente" \
: n'écris JAMAIS les mots "accélération", "accéléré", "ralentissement" ou "ralenti" au sujet de la \
cicatrisation ou de sa vitesse, y compris dans "Analyse quantitative" où seule une différence de \
taux moyens entre intervalles peut être mentionnée (jamais qu'un taux "accélère" ou "ralentit").
"""


# Appel isolé, séparé de PROMPT_RAPPORT : cette section a provoqué à deux reprises des
# fabrications de chiffres dans les sections numériques quand elle partageait le même appel
# LLM que les mesures (patron ouvert à trous, puis renforcement en prose -- voir
# chatbot_prompt_fragility en mémoire projet et "Fiabilité de la génération par LLM" dans le
# README). PROMPT_CONTEXTE ne reçoit jamais aucune mesure : la contamination croisée devient
# structurellement impossible, pas seulement évitée par une consigne de prompt.
#
# Reformulé le 2026-09-11 (voir chatbot_prompt_fragility, mémoire projet) : l'ancienne version
# autorisait explicitement une branche "le corpus ne permet pas de mettre en contexte" -- inutile
# en pratique puisque le corpus est curé spécifiquement sur le scratch assay, donc les articles
# retournés éclairent toujours au moins la problématique générale. Isolation-testée (3 générations,
# ancien vs nouveau prompt) : l'ancien a violé sa propre limite de "3 phrases maximum" sur 3/3
# générations et a cité une étude par position ("la première étude...") sur 1/3 malgré la
# consigne ; le nouveau (limite reformulée en "EXACTEMENT 3 phrases" + interdiction explicite de
# citer par année/position ajoutée en tête) obtient 2/3 générations parfaites et 1/3 à 4 phrases
# (sans aucune régression de fond) -- gardé.
#
# Renforcé une seconde fois le 2026-09-11, même session : un second lot de 3 générations en
# production a montré 3/3 dépassements sévères de la limite de 3 phrases ET 3/3 généralisations
# biologiques non garanties par de simples résumés d'articles ("directionnalité des cellules et
# coordination entre elles" présentée comme un fait établi) -- risque relevé par un relecteur
# externe : ce paragraphe peut laisser croire que l'expérience mesure la trajectoire/directionnalité
# cellulaire, alors qu'elle ne mesure que la surface de la plaie. Isolation-testée (3 vs 3, prompt
# ci-dessus vs candidat ajoutant une consigne de périmètre explicite + reformulant le point 1 pour
# décrire ce que les articles PROPOSENT plutôt que ce qu'ils "établissent" comme fait général) : le
# candidat obtient 2/3 générations propres contre 0/3 pour la version précédente sur ce lot ; 1/3
# génération du candidat reste imparfaite (plusieurs paragraphes + re-citation des titres complets)
# mais reproduit quasi mot pour mot la phrase de périmètre demandée. Gardé : amélioration nette sur
# le défaut ciblé, aucune régression nouvelle par rapport aux défauts déjà connus.
#
# Renforcé une troisième fois le 2026-09-12 (retour utilisateur : le paragraphe décrit les
# articles mais ne dit jamais explicitement pourquoi ils sont pertinents POUR cette analyse-ci --
# lu comme "générique" en usage réel). Point 1 modifié pour exiger que l'unique phrase générale se
# termine par une précision du type "ces travaux apportent un contexte méthodologique à l'analyse
# réalisée ici" (formulation libre, jamais quantitative). Isolation-testée (3 vs 3, avec le même
# contexte réel -- retrieval hybride, 3 articles de la catégorie analyse_temporelle) : sur la
# baseline, 1/3 génération contournait la règle "jamais d'année/position" via un tour de phrase non
# couvert par corriger_references_ordinales ("En premier, un article de 2021 propose..." -- pas
# "le premier article", donc jamais intercepté après coup), et aucune des 3 ne reliait
# explicitement le contexte à l'analyse en cours (lu comme trop générique, cohérent avec le retour
# utilisateur). Le candidat : 3/3 propres après le même post-traitement de troncature
# (limiter_paragraphe_contexte), 3/3 relient explicitement la littérature à "cette
# expérience"/"l'analyse réalisée ici" dès la phrase d'ouverture ou la 2e phrase (donc conservé même
# après troncature à 3 phrases), 0/3 mention d'année ou de position. Gardé : améliore l'objectif visé
# et n'introduit aucune régression -- au contraire, n'a pas reproduit le défaut de citation par année
# observé sur la baseline dans ce même lot.
PROMPT_CONTEXTE = """Tu es un assistant scientifique qui rédige la mise en contexte \
bibliographique d'un résumé d'expérience de cicatrisation de plaie (wound healing / scratch \
assay), à partir d'extraits d'articles déjà sélectionnés pour leur pertinence au domaine.

Extraits d'articles scientifiques pertinents :
{contexte}

Les titres de ces articles sont déjà affichés au lecteur juste avant ton paragraphe (ligne
"Articles considérés : ...") : NE LES RECITE PAS TOI-MÊME et NE LES NUMÉROTE JAMAIS (jamais de
titre entre guillemets, jamais "le premier article", jamais "l'article de [année]", jamais
"Titre 1"/"Titre 2"/"(Titre 1 et 2)"/"(Titre 1 à 3)" ni aucune autre référence par numéro ou
position) — ton paragraphe ne parle que de CE QU'ils établissent, jamais de leur identité ni de
leur rang, déjà connus du lecteur. Ne mentionne JAMAIS une année (2021, 2023, 2025...) ni un ordre
("le premier", "le second", "l'étude suivante") pour désigner un article ou une étude précise :
dis uniquement CE QUE les extraits montrent ENSEMBLE, jamais QUAND ni COMBIENTIÈME un point
précis a été montré.

Cette expérience mesure UNIQUEMENT la surface de la zone de plaie et son évolution temporelle
(fermeture, vitesse) : ne laisse JAMAIS entendre qu'elle mesure aussi la trajectoire, la vitesse
individuelle ou la directionnalité des cellules, même si les extraits ci-dessous en parlent — ce
sont des grandeurs propres à ces publications, pas à cette expérience.

Rédige un paragraphe COURT, EXACTEMENT 3 phrases au total, pas une de plus (même si plusieurs
extraits sont pertinents ou te semblent mériter un développement) — PAS une phrase ni un
paragraphe par article, ce paragraphe reste une synthèse groupée en 3 phrases seulement, jamais
une revue de littérature détaillée article par article, en suivant ces 3 phrases dans cet ordre :
1. Résume en UNE SEULE phrase (quitte à la construire en deux propositions reliées par une
   virgule ou "ainsi") ce que les extraits ci-dessus apportent à la compréhension GÉNÉRALE du
   scratch assay (intérêt de l'imagerie quantitative et de l'analyse temporelle pour caractériser
   la fermeture de la plaie et la migration cellulaire), en terminant cette même phrase par une
   précision du type "ces travaux apportent ainsi un contexte méthodologique à l'analyse réalisée
   ici" (formulation libre, mais sans jamais affirmer une concordance, une validation ou une
   comparaison chiffrée d'un résultat précis — ce lien reste uniquement méthodologique/thématique,
   jamais quantitatif) — décris CE QUE ces travaux PROPOSENT comme méthode ou objectif, jamais une
   conclusion biologique précise ("les cellules migrent de façon coordonnée", "leur mouvement est
   corrélé dans le temps"...) comme un fait général acquis : une méthode ou un objectif d'étude
   peut être résumé sans généraliser ses résultats spécifiques. Ces extraits ont déjà été
   sélectionnés pour leur rapport avec le scratch assay : n'écris JAMAIS qu'ils "ne sont pas
   pertinents" ou que le corpus "ne permet pas de contexte" — situe toujours, au minimum, la
   problématique générale qu'ils éclairent. Jamais de lien ni d'URL, jamais de détail
   méthodologique article par article.
2. N'affirme JAMAIS qu'un résultat de l'expérience (surface, %, vitesse) est "conforme à la
   littérature", "en accord avec les études", un "résultat normal", ou que les extraits le
   "confirment" : ne compare JAMAIS un chiffre de l'expérience à un chiffre lu dans un extrait,
   même approximativement. Tu n'as accès à aucune mesure de surface, de fermeture ni de vitesse :
   n'en mentionne ni n'en invente aucune, ce paragraphe ne parle que de ce que montrent les
   extraits ci-dessus.
3. Termine TOUJOURS par une phrase disant explicitement que le protocole, le système d'imagerie,
   les méthodes de quantification et les conditions biologiques de ces travaux diffèrent de cette
   expérience précise, et que les résultats obtenus ici ne sont donc pas directement comparables
   quantitativement à ceux rapportés dans ces publications.

N'invente jamais un mécanisme, une méthode ou un résultat absent des extraits ci-dessus.
"""


def _fr(nombre: float) -> str:
    """Formate un nombre avec une virgule décimale (convention française), plutôt
    que le point par défaut de Python (ex: 0.0h -> 0h, 1.58 -> 1,58) : ce texte
    est injecté tel quel dans les réponses du chatbot, en français."""
    if float(nombre) == int(nombre):
        return str(int(nombre))
    return str(nombre).replace(".", ",")


def _format_mesures(rows: list[dict]) -> str:
    lignes = []
    for r in rows:
        ligne = f"- T = {_fr(r['time_h'])}h : {r['area_px2']} px²"
        if "area_cm2" in r:
            ligne += f" ({_fr(r['area_cm2'])} cm²)"
        ligne += f", fermeture {_fr(r['closure_pct'])}% par rapport à T0"
        lignes.append(ligne)
    return "\n".join(lignes)


def _calculer_synthese(rows: list[dict]) -> dict:
    """Calcule les chiffres clés (surface initiale/finale, fermeture finale, durée,
    vitesse moyenne) directement dans le code plutôt que de laisser le LLM les
    déduire : un modèle local comme mistral 7B fait régulièrement des erreurs
    arithmétiques sur ce type de calcul (constaté : fermeture et vitesse moyenne
    erronées par rapport aux mesures fournies)."""
    t0, t_final = rows[0], rows[-1]
    duree_h = t_final["time_h"] - t0["time_h"]
    fermeture_finale_pct = t_final["closure_pct"]
    vitesse_moyenne_pct_h = round(fermeture_finale_pct / duree_h, 2) if duree_h > 0 else 0.0
    return {
        "t0": t0,
        "t_final": t_final,
        "duree_h": duree_h,
        "fermeture_finale_pct": fermeture_finale_pct,
        "vitesse_moyenne_pct_h": vitesse_moyenne_pct_h,
    }


def _format_synthese(synthese: dict, n_mesures: int) -> str:
    t0, t_final = synthese["t0"], synthese["t_final"]

    def _surface(r):
        s = f"{r['area_px2']} px²"
        if "area_cm2" in r:
            s += f" ({_fr(r['area_cm2'])} cm²)"
        return s

    return "\n".join([
        f"- Nombre de mesures : {n_mesures} (décrit une évolution globale, pas une cinétique fine)",
        f"- Surface initiale (T = {_fr(t0['time_h'])}h) : {_surface(t0)}",
        f"- Surface finale (T = {_fr(t_final['time_h'])}h) : {_surface(t_final)}",
        f"- Fermeture finale : {_fr(synthese['fermeture_finale_pct'])}% par rapport à T0",
        f"- Durée totale d'observation : {_fr(synthese['duree_h'])}h",
        f"- Vitesse moyenne de fermeture sur la période observée : {_fr(synthese['vitesse_moyenne_pct_h'])}%/h "
        "(points de % par heure, pas une vitesse instantanée)",
    ])


def _calculer_intervalles(rows: list[dict]) -> list[dict]:
    """Calcule la vitesse de fermeture entre chaque paire de points consécutifs.

    Sans ça, dire "la fermeture a ralenti entre 24h et 48h" oblige le LLM à comparer
    les points lui-même à partir des seules valeurs cumulées par rapport à T0, avec
    le même risque d'erreur d'interprétation que pour les chiffres globaux (constaté
    en pratique). En calculant la vitesse de chaque intervalle dans le code, le LLM
    n'a plus qu'à la commenter, pas à la déduire."""
    intervalles = []
    for prev, curr in zip(rows, rows[1:]):
        duree_h = curr["time_h"] - prev["time_h"]
        delta_closure_pct = round(curr["closure_pct"] - prev["closure_pct"], 1)
        vitesse_pct_h = round(delta_closure_pct / duree_h, 2) if duree_h > 0 else 0.0
        intervalles.append({
            "t_debut": prev["time_h"],
            "t_fin": curr["time_h"],
            "duree_h": duree_h,
            "delta_closure_pct": delta_closure_pct,
            "vitesse_pct_h": vitesse_pct_h,
        })
    return intervalles


def _format_intervalles(intervalles: list[dict]) -> str:
    if not intervalles:
        return "(un seul point de mesure disponible : pas d'intervalle à comparer)"

    lignes = []
    for iv in intervalles:
        if iv["delta_closure_pct"] < 0:
            # Delta négatif : la surface non colonisée a augmenté sur cet intervalle
            # (la plaie s'est partiellement rouverte), pas juste "fermé plus lentement" --
            # signalé explicitement ici plutôt que de laisser un "+" trompeur devant un
            # nombre négatif (ex: "+-19,9 points de %", corrigé le 2026-09-09).
            lignes.append(
                f"- Entre {_fr(iv['t_debut'])}h et {_fr(iv['t_fin'])}h : réouverture partielle, "
                f"la surface non colonisée a augmenté de {_fr(abs(iv['delta_closure_pct']))} points de %, "
                f"soit une vitesse de {_fr(iv['vitesse_pct_h'])}%/h sur cet intervalle"
            )
        else:
            lignes.append(
                f"- Entre {_fr(iv['t_debut'])}h et {_fr(iv['t_fin'])}h : fermeture +{_fr(iv['delta_closure_pct'])} "
                f"points de %, soit une vitesse de {_fr(iv['vitesse_pct_h'])}%/h sur cet intervalle"
            )
    if len(intervalles) > 1:
        plus_rapide = max(intervalles, key=lambda iv: iv["vitesse_pct_h"])
        plus_lent = min(intervalles, key=lambda iv: iv["vitesse_pct_h"])
        if plus_rapide is not plus_lent:
            lignes.append(
                f"- Intervalle le plus rapide : {_fr(plus_rapide['t_debut'])}h-{_fr(plus_rapide['t_fin'])}h "
                f"({_fr(plus_rapide['vitesse_pct_h'])}%/h) ; le plus lent : "
                f"{_fr(plus_lent['t_debut'])}h-{_fr(plus_lent['t_fin'])}h ({_fr(plus_lent['vitesse_pct_h'])}%/h)"
            )
    return "\n".join(lignes)


def _phrase_heterogeneite(synthese: dict, intervalles: list[dict]) -> str:
    """Phrase toute faite reliant le taux moyen global à l'hétérogénéité entre
    intervalles, composée dans le code plutôt que par le LLM : constaté en pratique,
    même avec les deux chiffres déjà disponibles séparément (taux moyen dans la
    synthèse, taux par intervalle dans _format_intervalles), un modèle local comme
    mistral 7B peut recopier le mauvais chiffre en recomposant cette phrase
    lui-même (ex: réutiliser le taux d'un intervalle à la place du taux global)."""
    if len(intervalles) < 2:
        return "(un seul intervalle disponible : pas de phrase de liaison pertinente)"

    plus_rapide = max(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    plus_lent = min(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    if plus_rapide is plus_lent:
        return "(taux identique sur tous les intervalles : pas d'hétérogénéité à signaler)"

    return (
        f"La vitesse moyenne de fermeture sur l'ensemble de la période "
        f"({_fr(synthese['vitesse_moyenne_pct_h'])}%/h) masque cette hétérogénéité : le taux "
        f"estimé est plus élevé sur l'intervalle {_fr(plus_rapide['t_debut'])}h-{_fr(plus_rapide['t_fin'])}h "
        f"({_fr(plus_rapide['vitesse_pct_h'])}%/h) que sur l'intervalle "
        f"{_fr(plus_lent['t_debut'])}h-{_fr(plus_lent['t_fin'])}h ({_fr(plus_lent['vitesse_pct_h'])}%/h)."
    )


def _phrase_liaison_resultats(synthese: dict, intervalles: list[dict]) -> str:
    """Phrase factuelle pré-composée en Python (jamais vue par le LLM), reliant le
    résultat chiffré de l'utilisateur au paragraphe "Mise en contexte scientifique"
    généré par PROMPT_CONTEXTE -- ajoutée APRÈS ce paragraphe, jamais à l'intérieur
    de son prompt.

    PROMPT_CONTEXTE reste structurellement aveugle aux mesures (voir sa docstring
    et chatbot_prompt_fragility, mémoire projet) : lui donner ces chiffres pour
    qu'il les intègre à sa prose reproduirait le mécanisme de fabrication déjà
    mesuré ailleurs dans ce projet (3/3 vitesses fabriquées quand mesures et
    contexte partageaient un seul appel). Cette phrase permet quand même au
    biologiste de relier ses résultats à la littérature qui précède, sans jamais
    exposer un seul chiffre à la génération libre du modèle."""
    t0, t_final = synthese["t0"], synthese["t_final"]
    base = (
        f"Dans les données analysées ici, la fermeture mesurée est de "
        f"{_fr(synthese['fermeture_finale_pct'])}% entre {_fr(t0['time_h'])}h et "
        f"{_fr(t_final['time_h'])}h"
    )
    if len(intervalles) < 2:
        return base + "."

    plus_rapide = max(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    plus_lent = min(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    if plus_rapide is plus_lent:
        return base + "."

    return (
        base + ", avec un taux estimé plus élevé sur l'intervalle "
        f"{_fr(plus_rapide['t_debut'])}h-{_fr(plus_rapide['t_fin'])}h "
        f"({_fr(plus_rapide['vitesse_pct_h'])}%/h) que sur l'intervalle "
        f"{_fr(plus_lent['t_debut'])}h-{_fr(plus_lent['t_fin'])}h "
        f"({_fr(plus_lent['vitesse_pct_h'])}%/h). Cette observation décrit uniquement "
        "les mesures disponibles, sans permettre à elle seule de conclure à une "
        "accélération biologique de la cicatrisation."
    )


def _phrase_interpretation(intervalles: list[dict]) -> str:
    """Phrase toute faite pour "Interprétation prudente", composée dans le code
    plutôt que laissée au LLM remplir un patron à trous. Constaté en pratique
    (2026-09-09) : même avec un patron explicite ("le taux ... est supérieur à
    celui ...") et une consigne de recopie stricte, mistral 7B fabrique parfois un
    signe "-" sur une vitesse pourtant positive, ou inverse l'ordre des bornes d'un
    intervalle -- un patron à trous laisse encore trop de place à l'erreur pour ce
    genre de comparaison numérique, contrairement à une phrase entièrement fournie
    (voir _phrase_heterogeneite, qui n'a jamais montré ce problème)."""
    if len(intervalles) < 2:
        return "(un seul intervalle disponible : pas de comparaison pertinente)"

    plus_rapide = max(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    plus_lent = min(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    if plus_rapide is plus_lent:
        return "(taux identique sur tous les intervalles : pas de comparaison pertinente)"

    return (
        f"Le taux moyen de fermeture estimé sur l'intervalle "
        f"{_fr(plus_rapide['t_debut'])}h-{_fr(plus_rapide['t_fin'])}h ({_fr(plus_rapide['vitesse_pct_h'])}%/h) "
        f"est supérieur à celui estimé sur l'intervalle "
        f"{_fr(plus_lent['t_debut'])}h-{_fr(plus_lent['t_fin'])}h ({_fr(plus_lent['vitesse_pct_h'])}%/h). "
        "Cette différence décrit uniquement les mesures disponibles et ne permet pas, à elle seule, "
        "de conclure à une accélération biologique de la cicatrisation."
    )


def _phrase_reouverture(intervalles: list[dict]) -> str:
    """Phrase toute faite signalant qu'un intervalle a un delta de fermeture négatif
    (la surface non colonisée a augmenté : la plaie s'est partiellement rouverte),
    composée dans le code plutôt que laissée au LLM. Constaté en pratique (retour du
    biologiste, 2026-09-09) : le patron d'"Interprétation prudente" de PROMPT_RAPPORT
    compare seulement "intervalle le plus rapide" vs "le plus lent", ce qui, pour un
    delta négatif, se traduit en une simple différence de vitesse -- masquant le fait
    plus important qu'il ne s'agit pas d'un ralentissement mais d'une réouverture
    partielle de la plaie, une information que le biologiste doit voir explicitement
    (mesure suspecte à vérifier, ou vrai phénomène biologique à creuser)."""
    reouvertures = [iv for iv in intervalles if iv["delta_closure_pct"] < 0]
    if not reouvertures:
        return "(aucune réouverture observée sur les intervalles disponibles)"

    lignes = [
        f"la surface non colonisée a augmenté de {_fr(abs(iv['delta_closure_pct']))} points de % "
        f"entre {_fr(iv['t_debut'])}h et {_fr(iv['t_fin'])}h"
        for iv in reouvertures
    ]
    return (
        "Réouverture partielle détectée sur les mesures disponibles : " + " ; ".join(lignes) + ". "
        "Cela peut correspondre à un vrai phénomène biologique (ex: décollement cellulaire) ou à "
        "une variation de mesure (segmentation, éclairage) : ce résumé ne permet pas de trancher, "
        "une vérification de l'image concernée est recommandée."
    )


# Correction après coup du placeholder "(aucune réouverture observée sur les intervalles
# disponibles)" que PROMPT_RAPPORT injecte pour signaler au LLM qu'il n'y a rien à ajouter sur
# ce point -- mistral 7B le recopie littéralement dans "Interprétation prudente" au lieu de
# l'ignorer (observé en usage réel, 2026-09-11 : 2/3 générations sur le cas sans réouverture).
# Deux corrections de prompt ont été testées en isolation et rejetées : renforcer la consigne
# ("ne la reprends pas") n'a rien amélioré (3/3 recopient encore) ; omettre le bloc entier du
# prompt quand il n'y a rien à signaler a provoqué une régression bien pire -- 2/3 générations
# ont alors FABRIQUÉ une réouverture inexistante ("la surface non colonisée a augmenté (réouverture
# partielle)" sur un jeu de mesures pourtant strictement décroissant). Voir chatbot_prompt_fragility,
# mémoire projet. Corrigé à la place en code pur, sur le texte déjà généré : aucun risque de
# fabrication puisque ça ne fait que supprimer une chaîne de caractères connue, jamais en ajouter.
#
# Le LLM ne recopie pas toujours le placeholder seul : il l'entoure parfois d'un
# amorce ("Il n'y a...") et/ou d'une chute ("... n'est détectée") pour en faire sa
# propre phrase. Le retirer seul laissait alors un fragment cassé en usage réel --
# "- Il n'y a" (amorce sans chute, 2026-09-12) et "et qu'n'est détectée." (amorce
# ET chute, 2026-09-12, voir chatbot_prompt_fragility en mémoire projet) : la regex
# absorbe donc aussi ces tournures courantes autour du placeholder, pas seulement le
# placeholder lui-même.
_PLACEHOLDER_REOUVERTURE_RE = re.compile(
    r"[ \t]*(?:\b(?:il\s+n['’]y\s+a(?:\s+pas)?|et\s+qu['’]|sans\s+qu['’]|de\s+plus,?)\s*)?"
    r"\(?\s*aucune r[ée]ouverture observ[ée]e sur les intervalles disponibles\s*\)?"
    r"(?:\s*n['’]est\s+(?:pas\s+)?d[ée]tect[ée]e?)?"
    r"\.?",
    re.IGNORECASE,
)


def nettoyer_placeholder_reouverture(texte: str) -> str:
    """Retire toute recopie littérale (avec ou sans parenthèses, en phrase autonome ou en
    milieu de paragraphe, avec ou sans amorce/chute ajoutée par le LLM) du placeholder
    "aucune réouverture observée..." que le LLM peut laisser fuiter dans le résumé final --
    ce texte est destiné au prompt uniquement, jamais à l'affichage. Ne modifie rien
    d'autre : si le placeholder n'apparaît pas dans `texte`, le retourne inchangé."""
    if "aucune réouverture" not in texte.lower() and "aucune reouverture" not in texte.lower():
        return texte
    texte = _PLACEHOLDER_REOUVERTURE_RE.sub("", texte)
    # Ligne devenue une puce vide (ne contenait que le placeholder).
    texte = re.sub(r"(?m)^[ \t]*-[ \t]*$\n?", "", texte)
    # Espace résiduel laissé par une suppression en milieu de phrase ou en fin de ligne.
    texte = re.sub(r"[ \t]{2,}", " ", texte)
    texte = re.sub(r"[ \t]+\n", "\n", texte)
    # Sauts de ligne excédentaires laissés par la puce supprimée (au plus un paragraphe vide).
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte.strip()


def generer_resume_stream(rows: list[dict], n_articles: int = 3):
    """Génère un résumé scientifique (style "Résultats") de l'évolution de la surface
    de la plaie au cours du temps, appuyé sur la littérature, en streaming.

    rows: sortie de kinetics.compute_kinetics_from_images (liste de dict avec au
    moins time_h, area_px2, closure_pct, triée par temps croissant).

    Retourne (générateur de morceaux de texte, sources).
    """
    # Reformulée le 2026-09-08 (voir chatbot_prompt_fragility, mémoire projet) : la
    # requête d'origine ("wound healing scratch assay closure rate over time cell
    # migration") ne retrouvait aucun article de la catégorie analyse_temporelle
    # dans le top 10, forçant systématiquement la section "Mise en contexte
    # scientifique" à répondre qu'elle ne peut rien dire -- honnête, mais peu
    # informatif. Ce vocabulaire, repris des résumés d'articles ciblés, retrouve
    # "Study of Wound Healing Dynamics by Single Pseudo-Particle Tracking..." au
    # rang 2. Isolation-testée 3/3 propre, gardée.
    #
    # Une 2e itération a aussi été essayée (vocabulaire repris de "The Frequent
    # Sampling of Wound Scratch Assay...", qui plaçait CE second article et le
    # précédent aux rangs 1 et 2 -- meilleur retrieval sur le papier) mais 3/3
    # générations ont alors fabriqué la vitesse moyenne et les vitesses par
    # intervalle (ex: 7,2%/h puis 0%/h au lieu de 0,83%/h et 1,58%/h) : cet article
    # contient lui-même beaucoup de chiffres de vitesse/cinétique dans son propre
    # résumé (fenêtre de 6h, doses de drogues, % de diminution de vitesse), ce qui
    # semble amener le LLM à les confondre avec les données de l'utilisateur.
    # Non gardée -- un meilleur retrieval n'implique pas une génération plus fiable.
    requete_litterature = (
        "wound healing assay reproducibility standardisation kinetics time-lapse "
        "phase contrast tracking cell migration dynamics"
    )
    # hybride=True (décision révisée le 2026-09-10, voir chatbot_prompt_fragility,
    # mémoire projet) : le risque documenté ci-dessus (fabrication de chiffres,
    # 2026-09-08) datait de l'architecture à UN SEUL appel LLM, où le contexte
    # littéraire et les mesures partageaient le même prompt. Depuis la séparation
    # en deux appels (PROMPT_RAPPORT / PROMPT_CONTEXTE, voir plus bas), cet appel-ci
    # ne reçoit plus jamais aucune mesure -- retesté en isolation (3 générations) :
    # 0 fabrication numérique, et un article pertinent supplémentaire remonte de
    # façon reproductible ("A novel method for evaluating and visualizing scratch
    # wound healing assays using level-set and image sector analysis", 2025). Le
    # risque d'origine ne s'applique donc plus structurellement à cet appel précis.
    # `_construire_prompt` (chat interactif, PROMPT_TEMPLATE) reste en revanche
    # hybride=False : ce prompt-là mélange encore mesures et contexte dans un seul
    # appel, donc le risque d'origine s'y applique toujours tel quel.
    articles = chercher(requete_litterature, n=n_articles, hybride=True)

    synthese = _calculer_synthese(rows)
    intervalles = _calculer_intervalles(rows)
    prompt_resultats = PROMPT_RAPPORT.format(
        synthese=_format_synthese(synthese, n_mesures=len(rows)),
        intervalles=_format_intervalles(intervalles),
        phrase_heterogeneite=_phrase_heterogeneite(synthese, intervalles),
        phrase_interpretation=_phrase_interpretation(intervalles),
        phrase_reouverture=_phrase_reouverture(intervalles),
        mesures=_format_mesures(rows),
    )
    prompt_contexte = PROMPT_CONTEXTE.format(contexte=_format_contexte(articles))

    # Les deux appels sont lancés eagerly, AVANT le générateur ci-dessous (pas seulement pour
    # garder le contrat de repondre_stream -- lever RuntimeError immédiatement si Ollama est
    # injoignable -- mais aussi pour les faire tourner en parallèle plutôt que l'un après
    # l'autre). _stream_ollama démarre son thread producteur dès l'appel, avant même que son
    # générateur soit consommé : le second appel se met donc déjà en file d'attente côté Ollama
    # pendant que le premier génère, au lieu d'attendre que flux_resultats soit épuisé. Aucun
    # changement de contenu ni de prompt (prompt_contexte est identique) : seul le moment où
    # la requête démarre change, donc aucun test d'isolation n'est nécessaire. Gain réel si
    # Ollama sert les deux requêtes en parallèle (OLLAMA_NUM_PARALLEL), sans jamais dégrader le
    # temps total si le serveur les sérialise malgré tout -- la détection de blocage
    # (CHUNK_TIMEOUT_S) ne démarre son décompte qu'à la première itération du générateur
    # consommateur, pas à la création du thread producteur, donc rien ne se déclenche à tort
    # pendant que ce second appel patiente en file d'attente.
    flux_resultats = _stream_ollama(prompt_resultats)
    flux_contexte = _stream_ollama(prompt_contexte)

    def flux():
        yield from flux_resultats
        yield "\n\n**Mise en contexte scientifique**\n\n"
        # Titres affichés en dur (Python), jamais laissés à la charge du LLM : voir
        # _intro_articles -- une consigne de prompt seule ("cite au moins un titre")
        # ne suffisait pas à garantir leur présence de façon fiable.
        yield _intro_articles(articles) + " "
        # Ce prompt ne contient aucune mesure (voir PROMPT_CONTEXTE), donc rien de ce résumé
        # numérique ne peut être contaminé par la génération de cette section.
        #
        # Accumulé en un seul bloc plutôt que streamé morceau par morceau (contrairement
        # au reste du résumé) : limiter_paragraphe_contexte a besoin du paragraphe complet
        # pour repérer et préserver la phrase de clôture avant de tronquer le surplus (voir
        # sa docstring) -- coût UX mineur (cette section apparaît d'un bloc au lieu de mot
        # à mot), accepté au profit d'une longueur fiable.
        texte_contexte = "".join(flux_contexte)
        yield limiter_paragraphe_contexte(texte_contexte)
        # Phrase numérique ajoutée APRÈS le paragraphe du LLM, jamais dans son
        # prompt (voir _phrase_liaison_resultats) : relie le résultat mesuré à la
        # littérature qui précède sans jamais exposer un chiffre à la génération.
        yield " " + _phrase_liaison_resultats(synthese, intervalles)

    return flux(), [a["titre"] for a in articles]


if __name__ == "__main__":
    resultat_test = {"surface_px2": 12450, "fermeture_pct": 42.0, "duree_h": 48}
    question_test = "Cette vitesse de fermeture est-elle cohérente avec la littérature ?"
    resultat = repondre(resultat_test, question_test)
    print(resultat["reponse"])
    print("\nSources :", ", ".join(resultat["sources"]))
