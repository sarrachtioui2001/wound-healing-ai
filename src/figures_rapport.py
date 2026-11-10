"""Schémas du rapport (architecture générale, diagramme de Gantt), régénérables.

Usage:
    python -m src.figures_rapport
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

GRIS, BLEU, VERT, ORANGE, VIOLET = "#63666A", "#4C72B0", "#55A868", "#DD8452", "#8172B3"


def _boite(ax, x, y, l, h, couleur, texte, taille=15, gras=True):
    ax.add_patch(FancyBboxPatch((x + 0.06, y - 0.06), l, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#D9D9D9", ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), l, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=couleur, ec="#333333", lw=1.6, zorder=2))
    ax.text(x + l / 2, y + h / 2, texte, ha="center", va="center", color="white", fontsize=taille,
            fontweight="bold" if gras else "normal", zorder=3, linespacing=1.4)


def _fleche(ax, x1, y1, x2, y2, style="-", couleur="#333333"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=couleur, linestyle=style, mutation_scale=18))


def figure_architecture(chemin):
    fig, ax = plt.subplots(figsize=(20, 10.5))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    y, h = 7.7, 1.75
    boites = [
        (0.5, 2.5, GRIS, "Image de\nplaie"),
        (3.4, 3.6, BLEU, "Segmentation\nU-Net + ResNet34"),
        (7.4, 3.6, VERT, "Quantification\nsurface / % fermeture"),
        (11.4, 3.1, VERT, "Suivi temporel\n0h / 24h / 48h..."),
        (14.9, 3.6, ORANGE, "Interprétation\nLLM + RAG"),
    ]
    for x, l, c, t in boites:
        _boite(ax, x, y, l, h, c, t)
    for (x1, l1, *_), (x2, *_) in zip(boites, boites[1:]):
        _fleche(ax, x1 + l1, y + h / 2, x2, y + h / 2)

    _boite(ax, 2.85, 4.5, 4.6, 1.5, BLEU,
           "Ensemble de modèles (option)\n→ indicateur d'incertitude\n(désaccord entre modèles)", taille=12.5, gras=False)
    _fleche(ax, 5.2, y, 5.15, 6.0, style="--", couleur=BLEU)

    _boite(ax, 2.85, 1.55, 14.4, 1.85, VIOLET,
           "Couche présentation — Interface Streamlit\n"
           "(saisie image, correction manuelle du masque, résultats, écoute du résumé)", taille=15)
    for x in (9.2, 12.95, 16.7):
        _fleche(ax, x, y, x, 3.4)

    ax.legend(handles=[Patch(color=BLEU, label="Couche vision"),
                       Patch(color=VERT, label="Couche quantification (Python)"),
                       Patch(color=ORANGE, label="Couche interprétation (LLM + RAG)"),
                       Patch(color=VIOLET, label="Couche présentation (Streamlit)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=13)
    fig.savefig(chemin, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Phases du projet et leurs dates, reconstituées à partir de l'historique git et des dates des fichiers.
PHASES = [
    ("Compréhension et préparation des données", "2026-08-31", "2026-09-01", GRIS),
    ("Modélisation (entraînement U-Net)", "2026-09-01", "2026-09-04", BLEU),
    ("Évaluation (baseline, TTA, tests, graines)", "2026-09-04", "2026-09-07", BLEU),
    ("Corpus documentaire et RAG/LLM", "2026-09-06", "2026-09-12", VERT),
    ("Ensemble de modèles et carte de confiance", "2026-09-09", "2026-09-09", BLEU),
    ("Application Streamlit", "2026-09-07", "2026-09-12", ORANGE),
    ("MLOps (Docker, MLflow, CI)", "2026-09-12", "2026-09-12", VIOLET),
    ("Rédaction du rapport", "2026-09-13", "2026-09-28", GRIS),
    ("Synthèse vocale du résumé", "2026-09-28", "2026-09-28", ORANGE),
]


def figure_gantt(chemin):
    import matplotlib.dates as mdates
    from datetime import date, timedelta

    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, (nom, debut, fin, couleur) in enumerate(PHASES):
        d0 = date.fromisoformat(debut)
        d1 = date.fromisoformat(fin) + timedelta(days=1)
        ax.barh(i, (d1 - d0).days, left=mdates.date2num(d0), height=0.55, color=couleur, edgecolor="white")
    ax.set_yticks(range(len(PHASES)), [p[0] for p in PHASES], fontsize=11)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.grid(axis="x", color="#e4e3df")
    ax.set_axisbelow(True)
    for cote in ("top", "right", "left"):
        ax.spines[cote].set_visible(False)
    ax.set_xlabel("Date (2026)")
    fig.tight_layout()
    fig.savefig(chemin, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("outputs/figures", exist_ok=True)
    figure_architecture("outputs/figures/figure_architecture.png")
    print("outputs/figures/figure_architecture.png")
    figure_gantt("outputs/figures/figure_gantt.png")
    print("outputs/figures/figure_gantt.png")
