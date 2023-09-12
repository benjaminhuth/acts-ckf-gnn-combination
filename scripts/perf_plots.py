import ROOT
import os
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.size': 12})


def plotTEfficency(tefficency, ax, **errorbar_kwargs):
    th1 = tefficency.GetTotalHistogram()

    bins = [ i for i in range(th1.GetNbinsX()) if th1.GetBinContent(i) > 0.0 ]

    x = [ th1.GetBinCenter(i) for i in bins ]

    x_lo = [ th1.GetBinLowEdge(i) for i in bins ]
    x_width = [ th1.GetBinWidth(i) for i in bins ]
    x_hi = np.add(x_lo, x_width)
    x_err_lo = np.subtract(x, x_lo)
    x_err_hi = np.subtract(x_hi, x)

    y = [ tefficency.GetEfficiency(i) for i in bins ]
    y_err_lo = [ tefficency.GetEfficiencyErrorLow(i) for i in bins ]
    y_err_hi = [ tefficency.GetEfficiencyErrorUp(i) for i in bins ]

    ax.errorbar(x, y, yerr=(y_err_lo, y_err_hi), xerr=(x_err_lo, x_err_hi), **errorbar_kwargs)
    return ax

performance_ckf = ROOT.TFile.Open(snakemake.input[0])
performance_gnn_ckf = ROOT.TFile.Open(snakemake.input[1])
performance_proof_of_concept = ROOT.TFile.Open(snakemake.input[2])
performance_truth_kalman = ROOT.TFile.Open(snakemake.input[3])

fig, axes = plt.subplots(2,3,figsize=(18,10))

plot_keys = ['trackeff_vs_eta', 'duplicationRate_vs_eta', "fakerate_vs_eta", 'trackeff_vs_pT', 'duplicationRate_vs_pT', "fakerate_vs_pT"]

for ax, key in zip(axes.flatten(), plot_keys):
    plotTEfficency(performance_proof_of_concept.Get(key), ax, fmt="none", color="tab:green", label="proof of concept")
    plotTEfficency(performance_truth_kalman.Get(key), ax, fmt="none", color="tab:red", label="truth kalman")
    plotTEfficency(performance_ckf.Get(key), ax, fmt="none", color="tab:blue", label="CKF only")
    plotTEfficency(performance_gnn_ckf.Get(key), ax, fmt="none", color="tab:orange", label="GNN + CKF")

    if "_pT" in key:
        ax.set_xscale('log')
        ax.set_xlim(0.9e-1, 1.1e2)
        ax.set_xticks([0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("pT [GeV]")
    elif "_eta" in key:
        ax.set_xlabel("$\eta$")

    ax.set_title(key.replace('_', ' '))
    ax.set_ylim(0,1)
    ax.legend(loc="lower left")

fig.tight_layout()

fig.savefig(snakemake.output[0])
