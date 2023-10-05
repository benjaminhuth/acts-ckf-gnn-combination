import ROOT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

from utils import plotTEfficency

plt.rcParams.update({"font.size": 12})

seed_performance_ckf = ROOT.TFile.Open(snakemake.input[0])
seed_performance_gnn_plus_ckf = ROOT.TFile.Open(snakemake.input[1])
seed_performance_proof_of_concept = ROOT.TFile.Open(snakemake.input[2])


#############################
# Extract aggregate numbers #
#############################

number_keys = ["total_seeds", "total_seed_purity", "seed_efficiency", "seed_fakerate", "seed_duplicationrate", "avg_duplicate_seeds"]
df = pd.DataFrame({"key": number_keys})

def add_to_df(perf_file, label):
    df[label] = [ perf_file.Get(key)[0] for key in number_keys ]

add_to_df(seed_performance_ckf, "CKF only")
add_to_df(seed_performance_gnn_plus_ckf, "GNN+CKF")
add_to_df(seed_performance_proof_of_concept, "proof of concept")

df = df.T.copy()
df.columns = number_keys
df = df[ [False, True, True, True] ].copy()
df = df.astype({"total_seeds": int})
df = df.reset_index().rename(columns={"index": "workflow"})

df.to_csv(snakemake.output[0], index=False)

pd.set_option('display.float_format', lambda x: f"{x:.3f}")
print(df,"\n")

fig, ax = plt.subplots(figsize=(15,1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

df = df.applymap(lambda n: f"{n:.3f}" if type(n) == float else str(n)).copy()
table = ax.table(cellText=df.values, colLabels=df.columns, edges="open", bbox = [0, 0, 1, 1], colLoc='right', loc='center')
table.set_fontsize(12)

for (row, col), cell in table.get_celld().items():
  if row == 0:
    cell.set_text_props(fontproperties=FontProperties(weight='bold'))

fig.savefig(snakemake.output[1])

##########################
# Plot eff/dup vs pt/eta #
##########################

fig, axes = plt.subplots(2, 2, figsize=(18, 10))

plot_keys = [
    "trackeff_vs_eta",
    "trackeff_vs_pT",
    "duplicationRate_vs_eta",
    "duplicationRate_vs_pT",
]

for ax, key in zip(axes.flatten(), plot_keys):
    print("plot",key)
    plotTEfficency(
        seed_performance_proof_of_concept.Get(key),
        ax,
        fmt="none",
        color="tab:green",
        label="proof of concept",
    )
    plotTEfficency(
        seed_performance_ckf.Get(key), ax, fmt="none", color="tab:blue", label="CKF only"
    )
    plotTEfficency(
        seed_performance_gnn_plus_ckf.Get(key),
        ax,
        fmt="none",
        color="tab:orange",
        label="GNN + CKF",
    )

    if "_pT" in key:
        ax.set_xscale("log")
        ax.set_xlim(0.9e-1, 1.1e2)
        ax.set_xticks([0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("pT [GeV]")
    elif "_eta" in key:
        ax.set_xlabel("$\eta$")

    ax.set_title("Seeding " + key.replace("_", " ").replace("seed", ""))
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")

fig.tight_layout()

fig.savefig(snakemake.output[2])


