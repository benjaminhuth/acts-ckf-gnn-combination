import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pprint
import re
import math

from utils_timing_plots import ChainPlotter

gpu_file = snakemake.input[0]
try:
    cpu_file = snakemake.input[1]
except:
    print("No CPU timing available")
    cpu_file = None

plotter = ChainPlotter(gpu_file, cpu_file)

chain_names = {
    "gnn": "GNN-based",
    "gnncpu": "GNN-based (CPU)",
    "ckf": "standard CKF",
    "poc": "proof of concept",
    "ttk": "truth tracking kalman",
}

keys_to_plot = ["gnn", "poc", "ckf"]
print("Consider",keys_to_plot)

total_times = {}

for k in keys_to_plot:
    print("aggregate",k)
    total_times[k] = sum(plotter.data_src[k].iloc[list(plotter.algs[k].keys())].time_perevent_s)

print("total times:")
pprint.pprint(total_times)

lo_range = (0, 6)

# Ensure we get 5-ticks here
f = total_times["ckf"] // 5
hi_range = 5 * f - 1, 5 * (f + 1) + 1

# But only the closer one
if hi_range[1] - f > f - hi_range[0]:
    hi_range = (hi_range[0], math.ceil(total_times["ckf"]))
else:
    hi_range = (math.floor(total_times["ckf"]), hi_range[1])
print("high_range",hi_range)

# figsize=(2 * len(keys_to_plot), 5)
figsize=(7,5)
fig, (ax_hi, ax_lo) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=figsize,
    height_ratios=[hi_range[1] - hi_range[0], lo_range[1] - lo_range[0]],
)

for x, key in enumerate(keys_to_plot):
    plotter.plot_chain(key, ax_hi, x, 100, lo_range=lo_range)
    plotter.plot_chain(key, ax_lo, x, 0.25, lo_range=lo_range)

ax_lo.set_xticks(np.arange(len(keys_to_plot)))
ax_lo.set_xticklabels([chain_names[k] for k in keys_to_plot])

# Set ranges
ax_hi.set_ylim(*hi_range)

ticks = np.arange(0,1000,5)
ticks = ticks[ (ticks >= hi_range[0]) & (ticks <= hi_range[1]) ]
ax_hi.set_yticks(ticks)

ax_lo.set_ylim(*lo_range)
ax_lo.set_yticks(np.arange(lo_range[0] + 5, lo_range[1], 5))

# Make plot nicer
ax_hi.spines["bottom"].set_visible(False)
ax_hi.xaxis.tick_top()
ax_hi.tick_params(labeltop=False)

ax_lo.spines["top"].set_visible(False)
ax_lo.xaxis.tick_bottom()

# Add tilted lines
d = 0.5
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kwargs)
ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kwargs)

ax_lo.set_ylabel("time per event [s]")
ax_hi.set_title("Timing comparison of the chains")

if snakemake.config["plt_show"]:
    plt.show()

fig.tight_layout()
fig.savefig(snakemake.output[0])
