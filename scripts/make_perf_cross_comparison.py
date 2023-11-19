import ROOT
import os
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path

from utils import *

plt.rcParams.update({"font.size": 12})


plot_keys = [
    "trackeff_vs_eta",
    "duplicationRate_vs_eta",
    "fakerate_vs_eta",
    "trackeff_vs_pT",
    "duplicationRate_vs_pT",
    "fakerate_vs_pT",
]


if snakemake.params["with_pt"]:
    ncols = 2
else:
    ncols = 1
    plot_keys = plot_keys[:3]

fig, axes = subplots(ncols, 3, snakemake)

replace_dict = {
    "_": " ",
    "eta": "$\eta$",
    "pT": "$p_T$",
    "duplicationRate" : "duplication rate",
    "fakerate" : "fake rate",
    "trackeff" : "matching efficiency",
}

def replace_key_text(key):
    for a, b in replace_dict.items():
        key = key.replace(a, b)

    return key

colors = [
    c for _, c in zip(range(len(snakemake.input)), matplotlib.colors.TABLEAU_COLORS)
]

for ax, key in zip(axes.flatten(), plot_keys):
    for f, color in zip(snakemake.input, colors):
        perf = ROOT.TFile.Open(f)
        name = Path(f).parent.name
        name = name.replace("_no_c", " (no combinatorics)").replace("_", " ")
        plotTEfficency(perf.Get(key), ax, fmt="none", color=color, label=name)

    if "_pT" in key:
        ax.set_xscale("log")
        ax.set_xlim(0.9e0, 1.1e2)
        ax.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("$p_T$ [GeV]")
    elif "_eta" in key:
        ax.set_xlabel("$\eta$")
        ax.set_xlim(-3.5, 3.5)

    ax.set_title(replace_key_text(key))
    ax.set_ylim(0, 1)


axes.flatten()[0].legend(loc="lower left")

fig.tight_layout()

fig.savefig(snakemake.output[0])
