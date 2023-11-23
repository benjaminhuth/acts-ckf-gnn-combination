import ROOT
import os
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path

import utils
from utils import plot_perf_root_file

plt.rcParams.update({"font.size": 12})

plot_keys = [
    "trackeff_vs_eta",
    "duplicationRate_vs_eta",
    "fakerate_vs_eta",
    "trackeff_vs_pT",
    "duplicationRate_vs_pT",
    "fakerate_vs_pT",
]

if snakemake.params.with_pt:
    ncols = 2
else:
    ncols = 1
    plot_keys = plot_keys[:3]

fig, axes = utils.subplots(ncols, 3, snakemake)

if "colors" in vars(snakemake.params):
    colors = snakemake.params.colors
else:
    colors = [
        c for _, c in zip(range(len(snakemake.input)), matplotlib.colors.TABLEAU_COLORS)
    ]

if "labels" in vars(snakemake.params):
    labels = snakemake.params[1]
    assert len(labels) == len(snakemake.input)
else:
    labels = len(snakemake.input)*[None]

print("Labels", labels)
print("Colors", colors)

for f, color, label in zip(snakemake.input, colors, labels):
    perf = ROOT.TFile.Open(f)
    if label is None:
        label = Path(f).parent.name

    label = label.replace("_", " ")
    plot_perf_root_file(axes, plot_keys, perf, label, color, eff_min=0.5, fake_max=0.5)

axes.flatten()[0].legend(loc="lower left")

fig.tight_layout()

fig.savefig(snakemake.output[0])
