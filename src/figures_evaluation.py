"""Graphiques du Chapitre 4 (évaluation), générés à partir des fichiers de résultats.

Usage:
    python -m src.figures_evaluation
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRED = "outputs/predictions"
OUT = "outputs/figures"

# Palette catégorielle validée (ordre fixe) : une couleur par méthode, partout.
COULEURS = {"Baseline": "#eb6834", "U-Net": "#2a78d6", "U-Net + TTA": "#1baf7a", "Ensemble (4)": "#eda100"}
ENCRE, ENCRE_2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRILLE, "axes.labelcolor": ENCRE_2, "xtick.color": ENCRE_2,
    "ytick.color": ENCRE_2, "axes.titlecolor": ENCRE, "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRILLE,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _lire(nom):
    with open(os.path.join(PRED, nom), encoding="utf-8") as f:
        return json.load(f)


def _sauver(fig, nom):
    chemin = os.path.join(OUT, nom)
    fig.savefig(chemin, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(chemin)


def figure_metriques():
    res = {
        "Baseline": _lire("test_baseline_metrics.json"),
        "U-Net": _lire("test_metrics.json"),
        "U-Net + TTA": _lire("test_metrics_tta.json"),
        "Ensemble (4)": _lire("test_ensemble_metrics.json"),
    }
    metriques = [("mean_dice", "Dice"), ("mean_iou", "IoU"), ("mean_precision", "Précision"), ("mean_recall", "Rappel")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.3, 1]})

    largeur = 0.2
    x = np.arange(len(metriques))
    for i, (methode, d) in enumerate(res.items()):
        valeurs = [d[k] for k, _ in metriques]
        barres = ax1.bar(x + (i - 1.5) * largeur, valeurs, largeur * 0.9, color=COULEURS[methode], label=methode)
        for b, v in zip(barres, valeurs):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}".replace(".", ","), ha="center",
                     va="bottom", fontsize=8.5, color=ENCRE_2)
    ax1.set_xticks(x, [n for _, n in metriques])
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Score moyen (14 images de test)")
    ax1.set_title("Recouvrement : plus haut = meilleur")
    ax1.grid(axis="x", visible=False)
    ax1.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))

    noms = list(res)
    erreurs = [res[m]["mean_area_error_pct"] for m in noms]
    y = np.arange(len(noms))[::-1]
    ax2.barh(y, erreurs, 0.6, color=[COULEURS[m] for m in noms])
    for yi, v in zip(y, erreurs):
        ax2.text(v + 6, yi, f"{v:.1f} %".replace(".", ","), va="center", fontsize=10, color=ENCRE)
    ax2.set_yticks(y, noms)
    ax2.set_xlim(0, max(erreurs) * 1.22)
    ax2.set_xlabel("Erreur de surface relative moyenne (%)")
    ax2.set_title("Erreur de surface : plus bas = meilleur")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    _sauver(fig, "figure6_comparaison_methodes.png")


def figure_dice_par_image():
    unet = {p["id"]: p["dice"] for p in _lire("test_metrics.json")["per_image"]}
    base = {p["id"]: p["dice"] for p in _lire("test_baseline_metrics.json")["per_image"]}
    ids = sorted(unet, key=lambda i: unet[i])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ids))
    for xi, i in zip(x, ids):
        ax.plot([xi, xi], [base[i], unet[i]], color="#b9b8b2", lw=2, zorder=1)
    ax.scatter(x, [base[i] for i in ids], s=70, color=COULEURS["Baseline"], edgecolor="white", lw=2, zorder=2,
               label=f"Baseline (moyenne {np.mean(list(base.values())):.3f})".replace(".", ","))
    ax.scatter(x, [unet[i] for i in ids], s=70, color=COULEURS["U-Net"], edgecolor="white", lw=2, zorder=3,
               label=f"U-Net (moyenne {np.mean(list(unet.values())):.3f})".replace(".", ","))
    ax.set_xticks(x, [str(i) for i in ids])
    ax.set_xlabel("Image de test (triées par Dice du U-Net)")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.02)
    ax.set_title("Dice image par image : le U-Net est meilleur sur les 14 images (Wilcoxon p = 0,000122)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    _sauver(fig, "figure7_dice_par_image.png")


def figure_robustesse_seeds():
    runs = [("Référence", _lire("test_metrics.json"))] + [
        (f"Graine {s}", _lire(f"seed{s}/test_metrics.json")) for s in (42, 123, 2026)
    ]
    noms = [n for n, _ in runs]
    dice = [d["mean_dice"] for _, d in runs]
    haus = [d["mean_hausdorff_normalized"] for _, d in runs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, valeurs, titre, fmt in ((ax1, dice, "Dice : stable d'un entraînement à l'autre", "{:.3f}"),
                                   (ax2, haus, "Hausdorff normalisé : très variable", "{:.3f}")):
        barres = ax.bar(noms, valeurs, 0.55, color=COULEURS["U-Net"])
        for b, v in zip(barres, valeurs):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v).replace(".", ","), ha="center", va="bottom",
                    fontsize=10, color=ENCRE)
        ax.set_title(titre)
        ax.grid(axis="x", visible=False)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Dice moyen (test)")
    ax2.set_ylim(0, max(haus) * 1.2)
    ax2.set_ylabel("Hausdorff normalisé moyen (test)")
    fig.tight_layout()
    _sauver(fig, "figure8_robustesse_seeds.png")


def figure_incertitude():
    ens = _lire("test_ensemble_metrics.json")
    pts = ens["per_image"]
    x = [p["pct_zone_incertaine"] for p in pts]
    y = [p["dice"] for p in pts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(25, color=ENCRE_2, lw=1, ls="--")
    ax.text(25.6, 0.955, "seuil d'alerte\nde l'application (25 %)", color=ENCRE_2, fontsize=9.5, va="top")
    ax.scatter(x, y, s=80, color=COULEURS["U-Net"], edgecolor="white", lw=2, zorder=3)
    for p in pts:
        if str(p["id"]) in ("6", "91", "93"):
            ax.annotate(f"image {p['id']}", (p["pct_zone_incertaine"], p["dice"]), textcoords="offset points",
                        xytext=(8, 6), fontsize=9.5, color=ENCRE)
    a, b = np.polyfit(x, y, 1)
    xs = np.linspace(min(x), max(x), 50)
    ax.plot(xs, a * xs + b, color="#9a9994", lw=2, zorder=2)
    r = ens["correlation_incertitude_dice"]
    ax.set_title(f"Plus les modèles sont en désaccord, plus le Dice baisse (r = {r:.3f})".replace(".", ","))
    ax.set_xlabel("Zone incertaine (% de la zone détectée où les 4 modèles divergent)")
    ax.set_ylabel("Dice de l'ensemble")
    fig.tight_layout()
    _sauver(fig, "figure9_incertitude_vs_dice.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure_metriques()
    figure_dice_par_image()
    figure_robustesse_seeds()
    figure_incertitude()
