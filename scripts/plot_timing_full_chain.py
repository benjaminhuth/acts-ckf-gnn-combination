import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pprint
import re

from utils_timing_plots import ChainPlotter


plotter = ChainPlotter(snakemake.input[0], snakemake.input[1])

chain_names = {
    "gnn": "GNN-based (GPU)",
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

f = total_times["ckf"] // 5
hi_range = 5 * f - 1, 5 * (f + 1) + 1


figsize=(2 * len(keys_to_plot), 5)
fig, (ax_hi, ax_lo) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=figsize,
    height_ratios=[hi_range[1] - hi_range[0], lo_range[1] - lo_range[0]],
)

for x, key in enumerate(keys_to_plot):
    plotter.plot_chain(key, ax_hi, x, 1000, lo_range=lo_range)
    plotter.plot_chain(key, ax_lo, x, lo_range=lo_range)

ax_lo.set_xticks(np.arange(len(keys_to_plot)))
ax_lo.set_xticklabels([chain_names[k] for k in keys_to_plot])

# Set ranges
ax_hi.set_ylim(*hi_range)
ax_hi.set_yticks([f * 5, (f + 1) * 5])

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
