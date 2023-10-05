import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd



timing_gpu = pd.read_csv(snakemake.input[0], sep='\t')
timing_cpu = pd.read_csv(snakemake.input[1], sep='\t')

pred = lambda i: "writer" in i.lower() or "reader" in i.lower()
timing_gpu = timing_gpu[ ~timing_gpu.identifier.apply(pred) ].reset_index(drop=True).copy()
timing_cpu = timing_cpu[ ~timing_cpu.identifier.apply(pred) ].reset_index(drop=True).copy()

print("# with GPU #")
print(timing_gpu)
print("-------------------------------")

print("# with CPU #")
print(timing_cpu)
print("-------------------------------")

# Chain names
chain_names = {
    "gnn": "GNN-based (GPU)",
    "gnncpu": "GNN-based (CPU)",
    "ckf": "standard CKF",
    "poc": "proof of concept",
    "ttk": "truth tracking kalman",
}

# X-Axis
x = {
    "gnn": 0, # main GNN-based pipeline
    "gnncpu": 1, # main GNN-pipeline on CPU
    "ckf": 2, # standard CKF
    "poc": 3, # proof of concept
    "ttk": 4, # truth tracking kalman
}

# Chain
algs = {
    "ckf": {7: "SeedingAlgorithm", 8: "TrackParamsEstimationAlgorithm", 10: "TrackFindingAlgorithm"},
    "poc": {12: "TruthTrackFinder", 13: "PrototracksToParsAndSeeds", 14: "CkfFromProtoTracks"},
    "gnn": {17: "TrackFindingMLBasedAlgorithm", 18: "PrototracksToParsAndSeeds", 19: "CkfFromProtoTracks"},
    "gnncpu": {6: "TrackFindingMLBasedAlgorithm", 7: "PrototracksToParsAndSeeds", 8: "CkfFromProtoTracks"},
    "ttk": {22: "TruthSeedingAlgorithm", 23: "TruthTrackFinder", 24: "TrackParamsEstimationAlgorithm", 25: "TrackFittingAlgorithm" },
}

cmaps = {
    "ckf": "Blues",
    "poc": "Reds",
    "gnn": "Greens",
    "gnncpu": "Greens",
    "ttk": "Purples",
}

data_src = {
    "ckf": timing_gpu,
    "poc": timing_gpu,
    "gnn": timing_gpu,
    "gnncpu": timing_cpu,
    "ttk": timing_gpu,
}

lo_range = (0, 10)

#hi_val = (max(timing_gpu.time_perevent_s) // 10) * 10 + 15
hi_range = (95, 101) #(hi_val - 10, hi_val)

#mi_val = (max(timing_cpu.time_perevent_s) // 10) * 10 + 10
mi_range = (60,70)


def plot_chain(key, ax, text_height_threshold = 0.5):
    y = 0
    
    n_algs = len(algs[key])
    colors = matplotlib.colormaps[cmaps[key]](np.linspace(0.5, 0.7, n_algs))
    timing = data_src[key]
    
    for (i, name), color in zip(algs[key].items(), colors):
        assert name in timing.iloc[i].identifier

        if name[-9:] == "Algorithm":
            name = name[:-9]

        t = timing.iloc[i].time_perevent_s
        
        bar = ax.bar(x[key], height=t, bottom=y, color=color).patches[0]

        if bar.get_height() > text_height_threshold:
            ytext = min(y + 0.5*t, y + 0.5*(lo_range[1] - y))
            ax.text(x[key], ytext, name, ha="center", va="center")

        
        y += t

fig, (ax_hi, ax_mi, ax_lo) = plt.subplots(
    3, 1, sharex=True, figsize=(10,5),
    height_ratios=[ hi_range[1]-hi_range[0], mi_range[1]-mi_range[0], lo_range[1]-lo_range[0] ]
)

for key in chain_names.keys():
    plot_chain(key, ax_hi, 1000)
    plot_chain(key, ax_mi, 1000)
    plot_chain(key, ax_lo)

ax_lo.set_xticks(np.arange(len(chain_names)))
ax_lo.set_xticklabels(chain_names.values())

# Set ranges
ax_hi.set_ylim(*hi_range)
ax_hi.set_yticks(np.arange(hi_range[0]+5, hi_range[1], 5))

ax_mi.set_ylim(*mi_range)
ax_mi.set_yticks(np.arange(mi_range[0]+5, mi_range[1], 5))

ax_lo.set_ylim(*lo_range)
ax_lo.set_yticks(np.arange(lo_range[0]+5, lo_range[1], 5))

# Make plot nicer
ax_hi.spines['bottom'].set_visible(False)
ax_hi.xaxis.tick_top()
ax_hi.tick_params(labeltop=False)

ax_lo.spines['top'].set_visible(False)
ax_lo.xaxis.tick_bottom()

ax_mi.spines['top'].set_visible(False)
ax_mi.spines['bottom'].set_visible(False)
ax_mi.tick_params(top=False, bottom=False)

# Add tilted lines
d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kwargs)
ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kwargs)
ax_mi.plot([0, 1], [0, 0], transform=ax_mi.transAxes, **kwargs)
ax_mi.plot([0, 1], [1, 1], transform=ax_mi.transAxes, **kwargs)

ax_mi.set_ylabel("time per event [s]")
ax_hi.set_title("Timing comparison of the chains")

if snakemake.config["plt_show"]:
    plt.show()
    
fig.tight_layout()
fig.savefig(snakemake.output[0])

#################
# Only pipeline #
#################

fig, ax = plt.subplots(figsize=(3,3))
plot_chain("gnn", ax, 0.00)


ax.set_xticks([x["gnn"]])
ax.set_xticklabels([chain_names["gnn"]])

ax.set_ylabel("time per event [s]")
ax.set_title("Timing of GNN based chain")

fig.tight_layout()
fig.savefig(snakemake.output[1])

