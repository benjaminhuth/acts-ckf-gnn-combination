import ROOT
import os
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import utils
from utils import plotTEfficency

color_dict = {
    "performance_proof_of_concept.root": "tab:green",
    "performance_truth_kalman.root": "tab:red",
    "performance_gnn_plus_ckf.root": "tab:orange",
    "performance_standard_ckf.root": "tab:blue",
}

plot_keys = [
    "trackeff_vs_eta",
    "duplicationRate_vs_eta",
    "fakerate_vs_eta",
    "trackeff_vs_pT",
    "duplicationRate_vs_pT",
    "fakerate_vs_pT",
]

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

if snakemake.params["with_pt"]:
    ncols = 2
else:
    ncols = 1
    plot_keys = plot_keys[:3]

fig, axes = utils.subplots(ncols, 3, snakemake)


for input_file in snakemake.input:
    performance_file = ROOT.TFile.Open(input_file)

    name = Path(input_file).name
    label = name.replace("_", " ").replace("plus", "&").replace(".root", "").replace("performance", "").strip()
    color = color_dict[name]

    for ax, key in zip(axes.flatten(), plot_keys):
        no_yerr=True if "kalman" in name and "pT" in key else False
        plotTEfficency(
            performance_file.Get(key),
            ax,
            no_yerr=no_yerr,
            fmt="none",
            color=color,
            label=label,
        )

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

        # if "fake" in key:
        #     ax.legend(loc="upper right")
        # else:
        #     ax.legend(loc="lower left")

# for ax in axes.flatten():
#     if "dup" in ax.get_title() or "fake" in ax.get_title():
#         ax.set_yscale('log')
#         ax.set_ylim(1e-2, 1)

axes.flatten()[0].legend(loc="lower left")

fig.tight_layout()
fig.savefig(snakemake.output[0])
