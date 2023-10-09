import ROOT
import os
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path

from utils import plotTEfficency

plt.rcParams.update({"font.size": 12})

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

plot_keys = [
    "trackeff_vs_eta",
    "duplicationRate_vs_eta",
    "fakerate_vs_eta",
    "trackeff_vs_pT",
    "duplicationRate_vs_pT",
    "fakerate_vs_pT",
]

colors = [
    c for _, c in zip(range(len(snakemake.input)), matplotlib.colors.TABLEAU_COLORS)
]

for ax, key in zip(axes.flatten(), plot_keys):
    for f, color in zip(snakemake.input, colors):
        perf = ROOT.TFile.Open(f)
        name = Path(f).parent.name.replace("_", " ").strip()

        plotTEfficency(perf.Get(key), ax, fmt="none", color=color, label=name)

    if "_pT" in key:
        ax.set_xscale("log")
        ax.set_xlim(0.9e-1, 1.1e2)
        ax.set_xticks([0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("pT [GeV]")
    elif "_eta" in key:
        ax.set_xlabel("$\eta$")

    ax.set_title(key.replace("_", " "))
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")

fig.tight_layout()

fig.savefig(snakemake.output[0])
