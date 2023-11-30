import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from utils_timing_plots import ChainPlotter, plot_stages


for chain, output in zip(["gnn", "gnn-nc"], snakemake.output[:2]):
    fig, ax = plt.subplots(figsize=(7,5))

    for x, f1, f2 in zip([0,1], snakemake.input[0::2], snakemake.input[1::2]):
        plotter = ChainPlotter(f1)
        ys = plotter.plot_chain(chain, ax, x, 10000)
        plot_stages(x, ax, f2, plotter)

        t = ys[-1] - ys[-2]
        ax.text(x, ys[-1]-0.5*t, f"Modified CKF ({t:.2f})", va="center", ha="center")

    ax.set_xticks([0,1])
    ax.set_xticklabels([ w.replace("_", " ") for w in snakemake.wildcards ])
    ax.set_ylabel("time per event [s]")
    ax.set_title("Timing of GNN based chains")

    fig.tight_layout()
    fig.savefig(output)


# Combine these
fig, ax = plt.subplots(figsize=(7,5))

for x, f1, f2 in zip([0,1], snakemake.input[0::2], snakemake.input[1::2]):
    plotter = ChainPlotter(f1)

    for chain, algname, disp in zip(
        ["gnn", "gnn-nc"],
        ["gpcCkfFromProtoTracks", "gpcncCkfFromProtoTracks"],
        [-0.203, +0.203],
    ):
        ys = plotter.plot_chain(chain, ax, x, 10000, displace_dict={algname: disp})
        print(chain,"ys", ys)
        t = ys[-1] - ys[-2]

        fitter = "MCKF\n(no comb.)" if "nc" in chain else "MCKF"

        ax.text(x-disp, ys[-1]-0.5*t, f"{fitter}\n{t:.2f}", va="center", ha="center")

    plot_stages(x, ax, f2, plotter)

ax.set_xticks([0,1])
ax.set_xticklabels([ w.replace("_", " ") for w in snakemake.wildcards ])
ax.set_ylabel("time per event [s]")
ax.set_title("Timing of GNN based chains")

fig.tight_layout()
fig.savefig(snakemake.output[2])
