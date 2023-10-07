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

performance_ckf = ROOT.TFile.Open(snakemake.input[0])
performance_gnn_ckf = ROOT.TFile.Open(snakemake.input[1])
performance_proof_of_concept = ROOT.TFile.Open(snakemake.input[2])
performance_truth_kalman = ROOT.TFile.Open(snakemake.input[3])

fig, axes = utils.subplots(2, 3, snakemake)

plot_keys = [
    "trackeff_vs_eta",
    "duplicationRate_vs_eta",
    "fakerate_vs_eta",
    "trackeff_vs_pT",
    "duplicationRate_vs_pT",
    "fakerate_vs_pT",
]

for ax, key in zip(axes.flatten(), plot_keys):
    plotTEfficency(
        performance_proof_of_concept.Get(key),
        ax,
        fmt="none",
        color="tab:green",
        label="proof of concept",
    )
    plotTEfficency(
        performance_truth_kalman.Get(key),
        ax,
        fmt="none",
        color="tab:red",
        label="truth kalman",
    )
    plotTEfficency(
        performance_ckf.Get(key), ax, fmt="none", color="tab:blue", label="CKF only"
    )
    plotTEfficency(
        performance_gnn_ckf.Get(key),
        ax,
        fmt="none",
        color="tab:orange",
        label="GNN + CKF",
    )

    if "_pT" in key:
        ax.set_xscale("log")
        ax.set_xlim(0.9e0, 1.1e2)
        ax.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("pT [GeV]")
    elif "_eta" in key:
        ax.set_xlabel("$\eta$")
        ax.set_xlim(-3.5, 3.5)

    ax.set_title(key.replace("_", " "))
    ax.set_ylim(0, 1)

    if "fake" in key:
        ax.legend(loc="upper right")
    else:
        ax.legend(loc="lower left")

fig.tight_layout()

fig.savefig(snakemake.output[0])
