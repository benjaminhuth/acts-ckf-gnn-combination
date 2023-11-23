import ROOT
import os
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import utils
from utils import plot_perf_root_file

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

if snakemake.params["with_pt"]:
    ncols = 2
else:
    ncols = 1
    plot_keys = plot_keys[:3]

print("Perf plots for keys:",plot_keys)

fig, axes = utils.subplots(ncols, 3, snakemake)
axes = axes.flatten()

assert len(axes) == len(plot_keys)

for input_file in snakemake.input:
    performance_file = ROOT.TFile.Open(input_file)

    name = Path(input_file).name
    label = name.replace("_", " ").replace("plus", "&").replace(".root", "").replace("performance", "").strip()
    color = color_dict[name]

    plot_perf_root_file(axes, plot_keys, performance_file, label, color, eff_min=0.5, fake_max=0.5)

axes[0].legend(loc="lower left")

fig.tight_layout()
fig.savefig(snakemake.output[0])
