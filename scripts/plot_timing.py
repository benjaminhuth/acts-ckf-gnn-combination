import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd



timing = pd.read_csv(snakemake.input[0], sep='\t')


print(timing)

# Chain names
chain_names = {
    "gnn": "GNN-based pipeline",
    "ckf": "standard CKF",
    "poc": "proof of concept",
    "ttk": "truth tracking kalman",
}

# X-Axis
x = {
    "gnn": 0, # main GNN-based pipeline
    "ckf": 1, # standard CKF
    "poc": 2, # proof of concept
    "ttk": 3, # truth tracking kalman
}

# Chain
algs = {
    "ckf": {11: "SeedingAlgorithm", 12: "TrackParamsEstimationAlgorithm", 14: "TrackFindingAlgorithm"},
    "poc": {17: "TruthTrackFinder", 18: "PrototracksToParsAndSeeds", 21: "CkfFromProtoTracks"},
    "gnn": {25: "TrackFindingMLBasedAlgorithm", 27: "PrototracksToParsAndSeeds", 30: "CkfFromProtoTracks"},
    "ttk": {37: "TruthTrackFinder", 38: "TrackParamsEstimationAlgorithm", 39: "TrackFittingAlgorithm" },
}

cmaps = {
    "ckf": "Blues",
    "poc": "Reds",
    "gnn": "Greens",
    "ttk": "Purples",
}

def plot_chain(key, ax):
    y = 0
    
    n_algs = len(algs[key])
    colors = matplotlib.colormaps[cmaps[key]](np.linspace(0.5, 0.7, n_algs))
    
    for (i, name), color in zip(algs[key].items(), colors):
        assert name in timing.iloc[i].identifier
        t = timing.iloc[i].time_perevent_s
        
        ax.bar(x[key], height=t, bottom=y, color=color)
        
        y += t

lo_range = (0, 30)
hi_range = (70, 85)

f, (ax_hi, ax_lo) = plt.subplots(2, 1, sharex=True, figsize=(8,5), 
                                 height_ratios=[ hi_range[1]-hi_range[0], lo_range[1]-lo_range[0] ])

for key in chain_names.keys():
    plot_chain(key, ax_hi)
    plot_chain(key, ax_lo)

ax_lo.set_xticks(np.arange(len(chain_names)))
ax_lo.set_xticklabels(chain_names.values())

# Set ranges
ax_hi.set_ylim(*hi_range)
ax_hi.set_yticks(np.arange(hi_range[0]+5, hi_range[1], 5))

ax_lo.set_ylim(*lo_range)
ax_lo.set_yticks(np.arange(lo_range[0]+5, lo_range[1], 5))

# Make plot nicer
ax_hi.spines['bottom'].set_visible(False)
ax_hi.xaxis.tick_top()
ax_hi.tick_params(labeltop=False)

ax_lo.spines['top'].set_visible(False)
ax_lo.xaxis.tick_bottom()

# Add tilted lines
d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kwargs)
ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kwargs)

if snakemake.config["plt_show"]:
    plt.show()
    
fig.savefig(snakemake.output[0])

