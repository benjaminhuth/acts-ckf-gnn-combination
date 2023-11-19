import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from utils_timing_plots import ChainPlotter, plot_stages


plotter = ChainPlotter(snakemake.input[0])

fig, ax = plt.subplots(figsize=(4,5))
plotter.plot_chain("gnn", ax, 0, 10000)

plot_stages(0, ax, snakemake.input[1], plotter)

ax.set_ylabel("time per event [s]")
ax.set_title("Timing of GNN based chain")

fig.tight_layout()
fig.savefig(snakemake.output[0])
