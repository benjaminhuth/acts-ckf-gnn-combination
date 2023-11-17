import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pprint
import re


timing_gpu = pd.read_csv(snakemake.input[0], sep="\t")
timing_cpu = pd.read_csv(snakemake.input[1], sep="\t")

pred = lambda i: "writer" in i.lower() or "reader" in i.lower()
timing_gpu = (
    timing_gpu[~timing_gpu.identifier.apply(pred)].reset_index(drop=True).copy()
)
timing_cpu = (
    timing_cpu[~timing_cpu.identifier.apply(pred)].reset_index(drop=True).copy()
)

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

# Chain
algs = {
    "ckf": {
        7: "SeedingAlgorithm",
        8: "TrackParamsEstimationAlgorithm",
        10: "TrackFindingAlgorithm",
    },
    "poc": {
        11: "TruthTrackFinder",
        12: "PrototracksToParsAndSeeds",
        13: "CkfFromProtoTracks",
    },
    "gnn": {
        15: "TrackFindingMLBasedAlgorithm",
        16: "PrototracksToParsAndSeeds",
        17: "CkfFromProtoTracks",
    },
    "gnncpu": {
        6: "TrackFindingMLBasedAlgorithm",
        7: "PrototracksToParsAndSeeds",
        8: "CkfFromProtoTracks",
    },
    "ttk": {
        22: "TruthSeedingAlgorithm",
        23: "TruthTrackFinder",
        24: "TrackParamsEstimationAlgorithm",
        25: "TrackFittingAlgorithm",
    },
}

cmaps = {
    "ckf": "Blues",
    "poc": "Greens",
    "gnn": "Oranges",
    "gnncpu": "Oranges",
    "ttk": "Purples",
}

data_src = {
    "ckf": timing_gpu,
    "poc": timing_gpu,
    "gnn": timing_gpu,
    "gnncpu": timing_cpu,
    "ttk": timing_gpu,
}

class RemapDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __missing__(self, key):
        return key

remap_names = RemapDict({
    "CkfFromProtoTracks": "Modified CKF",
    "TrackFindingMLBased": "GNN Pipeline",
    "TrackFinding" : "Standard CKF"
})


keys_to_plot = ["gnn", "poc", "ckf"]
print("Consider",keys_to_plot)

total_times = {}

for k in keys_to_plot:
    print("aggregate",k)
    total_times[k] = sum(data_src[k].iloc[list(algs[k].keys())].time_perevent_s)

print("total times:")
pprint.pprint(total_times)

lo_range = (0, 6)

f = total_times["ckf"] // 5
hi_range = 5 * f - 1, 5 * (f + 1) + 1

# mi_val = (max(timing_cpu.time_perevent_s) // 10) * 10 + 10
# mi_range = (60,70)


def plot_chain(key, ax, x, text_height_threshold=0.5):
    y = 0

    n_algs = len(algs[key])
    colors = matplotlib.colormaps[cmaps[key]](np.linspace(0.5, 0.7, n_algs))
    timing = data_src[key]

    for (i, name), color in zip(algs[key].items(), colors):
        assert name in timing.iloc[i].identifier, f"{name} not in {timing.iloc[i].identifier}"

        if name[-9:] == "Algorithm":
            name = name[:-9]

        t = timing.iloc[i].time_perevent_s

        bar = ax.bar(x, height=t, bottom=y, color=color).patches[0]

        if bar.get_height() > text_height_threshold:
            ytext = min(y + 0.5 * t, y + 0.5 * (lo_range[1] - y))
            ax.text(x, ytext, remap_names[name], ha="center", va="center")

        y += t


# fig, (ax_hi, ax_mi, ax_lo) = plt.subplots(
#     3, 1, sharex=True, figsize=(10,5),
#     height_ratios=[ hi_range[1]-hi_range[0], mi_range[1]-mi_range[0], lo_range[1]-lo_range[0] ]
# )

figsize=(2 * len(keys_to_plot), 5)
fig, (ax_hi, ax_lo) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=figsize,
    height_ratios=[hi_range[1] - hi_range[0], lo_range[1] - lo_range[0]],
)

for x, key in enumerate(keys_to_plot):
    plot_chain(key, ax_hi, x, 1000)
    # plot_chain(key, ax_mi, 1000)
    plot_chain(key, ax_lo, x)

ax_lo.set_xticks(np.arange(len(keys_to_plot)))
ax_lo.set_xticklabels([chain_names[k] for k in keys_to_plot])

# Set ranges
ax_hi.set_ylim(*hi_range)
ax_hi.set_yticks([f * 5, (f + 1) * 5])

# ax_mi.set_ylim(*mi_range)
# ax_mi.set_yticks(np.arange(mi_range[0]+5, mi_range[1], 5))

ax_lo.set_ylim(*lo_range)
ax_lo.set_yticks(np.arange(lo_range[0] + 5, lo_range[1], 5))

# Make plot nicer
ax_hi.spines["bottom"].set_visible(False)
ax_hi.xaxis.tick_top()
ax_hi.tick_params(labeltop=False)

ax_lo.spines["top"].set_visible(False)
ax_lo.xaxis.tick_bottom()

# ax_mi.spines['top'].set_visible(False)
# ax_mi.spines['bottom'].set_visible(False)
# ax_mi.tick_params(top=False, bottom=False)

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
# ax_mi.plot([0, 1], [0, 0], transform=ax_mi.transAxes, **kwargs)
# ax_mi.plot([0, 1], [1, 1], transform=ax_mi.transAxes, **kwargs)

ax_lo.set_ylabel("time per event [s]")
ax_hi.set_title("Timing comparison of the chains")

if snakemake.config["plt_show"]:
    plt.show()

fig.tight_layout()
fig.savefig(snakemake.output[0])

#################
# Only pipeline #
#################

fig, ax = plt.subplots(figsize=figsize)
plot_chain("gnn", ax, 0, 10000)

with open(snakemake.input[2], 'r') as f:
    graph_building_time = None
    classifier_times = []
    track_building_time = None

    gb_regex = r"^.*INFO\s+- graph building:\s+([0-9]+.[0-9]+)\s\+-\s([0-9]+(.[0-9]+)?)$"
    clf_regex = r"^.*INFO\s+- classifier:\s+([0-9]+.[0-9]+)\s\+-\s([0-9]+(.[0-9]+)?)$"
    tb_regex = r"^.*INFO\s+- track building:\s+([0-9]+.[0-9]+)\s\+-\s([0-9]+(.[0-9]+)?)$"

    for line in f:
        m = re.match(gb_regex, line)
        if m:
            assert graph_building_time is None
            print("Match graph building timing")
            graph_building_time = float(m[1]) / 1000

        m = re.match(clf_regex, line)
        if m:
            print("Match classifier timing")
            classifier_times.append(float(m[1]) / 1000)

        m = re.match(tb_regex, line)
        if m:
            assert track_building_time is None
            print("Match track building timing")
            track_building_time = float(m[1]) / 1000

assert graph_building_time is not None
assert track_building_time is not None
assert len(classifier_times) > 0

classifier_labels = [ f"GNN{i}" for i in range(len(classifier_times)) ]
classifier_labels[0]  = "MLP filter"

ax.set_xticks([0])
ax.set_xticklabels(["GNN-based (GPU)"])

time_pipeline = timing_gpu.iloc[list(algs["gnn"].keys())[0]].time_perevent_s
diff = time_pipeline - (graph_building_time + sum(classifier_times) + track_building_time)
gap_time = diff / (2+len(classifier_times)+1)
print("time pipeline:",time_pipeline,"diff",diff,"gap:",gap_time)

bar_args = dict(color='black', alpha=0.2, edgecolor=None, width=0.7)

s = gap_time

ax.bar(0, graph_building_time, bottom=s, **bar_args)
ax.text(0, s+graph_building_time/2, "graph building", ha="center", va="center")
s += (graph_building_time + gap_time)

for clf_time, label in zip(classifier_times, classifier_labels):
    ax.bar(0, clf_time, bottom=s, **bar_args)
    ax.text(0, s+clf_time/2, label, ha="center", va="center")
    s += (clf_time + gap_time)

ax.bar(0, track_building_time, bottom=s, **bar_args)
ax.text(0, s+track_building_time/2, "track building", ha="center", va="center")

ax.set_ylabel("time per event [s]")
ax.set_title("Timing of GNN based chain")

fig.tight_layout()
fig.savefig(snakemake.output[1])
