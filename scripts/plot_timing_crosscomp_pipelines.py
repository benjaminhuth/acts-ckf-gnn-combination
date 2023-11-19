import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from utils_timing_plots import ChainPlotter, plot_stages



fig, ax = plt.subplots(figsize=(9,7))

plotterA = ChainPlotter(snakemake.input[0])
plotterA.plot_chain("gnn", ax, 0, 10000)
plot_stages(0, ax, snakemake.input[1], plotterA)

plotterB = ChainPlotter(snakemake.input[2])
plotterB.plot_chain("gnn", ax, 1, 10000)
plot_stages(1, ax, snakemake.input[3], plotterB)

ax.set_xticks([0,1])
ax.set_xticklabels(snakemake.wildcards)
ax.set_ylabel("time per event [s]")
ax.set_title("Timing of GNN based chains")

fig.tight_layout()
fig.savefig(snakemake.output[0])
