import numpy as np
import pandas as pd
import matplotlib
import pprint
import re

class RemapDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __missing__(self, key):
        return key


class ChainPlotter:
    def __init__(self, timing_tsv_gpu=None, timing_tsv_cpu=None):
        pred = lambda i: "writer" in i.lower() or "reader" in i.lower()

        if timing_tsv_gpu is not None:
            self.timing_gpu = pd.read_csv(timing_tsv_gpu, sep="\t")
            self.timing_gpu = (
                self.timing_gpu[~self.timing_gpu.identifier.apply(pred)].reset_index(drop=True).copy()
            )
        else:
            self.timing_gpu = None

        if timing_tsv_cpu is not None:
            self.timing_cpu = pd.read_csv(timing_tsv_cpu, sep="\t")
            self.timing_cpu = (
                self.timing_cpu[~self.timing_cpu.identifier.apply(pred)].reset_index(drop=True).copy()
            )
        else:
            self.timing_cpu = None

        print("# Timings with GPU #")
        print(self.timing_gpu)
        print("-------------------------------")

        print("# Timings with CPU #")
        print(self.timing_cpu)
        print("-------------------------------")

        self.data_src = {
            "ckf": self.timing_gpu,
            "poc": self.timing_gpu,
            "gnn": self.timing_gpu,
            "gnn-nc": self.timing_gpu,
            "gnncpu": self.timing_cpu,
            "ttk": self.timing_gpu,
        }

        # Chain
        self.algs = {
            "ckf": {
                7: "SeedingAlgorithm",
                8: "TrackParamsEstimationAlgorithm",
                10: "TrackFindingAlgorithm",
            },
            "poc": {
                11: "TruthTrackFinder",
                12: "PrototracksToParsAndSeeds",
                13: "pocCkfFromProtoTracks",
            },
            "gnn": {
                15: "TrackFindingMLBasedAlgorithm",
                16: "PrototracksToParsAndSeeds",
                17: "gpcCkfFromProtoTracks",
            },
            "gnn-nc": {
                15: "TrackFindingMLBasedAlgorithm",
                16: "PrototracksToParsAndSeeds",
                20: "gpcncCkfFromProtoTracks",
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

        self.cmaps = {
            "ckf": "Blues",
            "poc": "Greens",
            "gnn": "Oranges",
            "gnn-nc": "Oranges",
            "gnncpu": "Oranges",
            "ttk": "Purples",
        }

        self.remap_names = RemapDict({
            "CkfFromProtoTracks": "Modified CKF",
            "TrackFindingMLBased": "GNN Pipeline",
            "TrackFinding" : "Standard CKF"
        })

    def plot_chain(self, key, ax, x, text_height_threshold=0.5, lo_range=None):
        y = 0

        n_algs = len(self.algs[key])
        colors = matplotlib.colormaps[self.cmaps[key]](np.linspace(0.5, 0.7, n_algs))
        timing = self.data_src[key]

        for (i, name), color in zip(self.algs[key].items(), colors):
            assert name in timing.iloc[i].identifier, f"{name} not in {timing.iloc[i].identifier}"

            t = timing.iloc[i].time_perevent_s

            bar = ax.bar(x, height=t, bottom=y, color=color).patches[0]

            if name[-9:] == "Algorithm":
                name = name[:-9]
            if "CkfFromProtoTracks" in name:
                name = "CkfFromProtoTracks"

            if bar.get_height() > text_height_threshold:
                ytext = min(y + 0.5 * t, y + 0.5 * (lo_range[1] - y))
                ax.text(x, ytext, self.remap_names[name], ha="center", va="center")

            y += t

        return ax


def plot_stages(x, ax, logfile, plotter):
    with open(logfile, 'r') as f:
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

    time_pipeline = plotter.timing_gpu.iloc[list(plotter.algs["gnn"].keys())[0]].time_perevent_s
    diff = time_pipeline - (graph_building_time + sum(classifier_times) + track_building_time)
    gap_time = diff / (2+len(classifier_times)+1)
    print("time pipeline:",time_pipeline,"diff",diff,"gap:",gap_time)

    bar_args = dict(color='black', alpha=0.2, edgecolor=None, width=0.7)

    s = gap_time

    ax.bar(x, graph_building_time, bottom=s, **bar_args)
    ax.text(x, s+graph_building_time/2, f"graph building ({graph_building_time:.2f})", ha="center", va="center")
    s += (graph_building_time + gap_time)

    for clf_time, label in zip(classifier_times, classifier_labels):
        ax.bar(x, clf_time, bottom=s, **bar_args)
        ax.text(x, s+clf_time/2, f"{label} ({clf_time:.2f})", ha="center", va="center")
        s += (clf_time + gap_time)

    ax.bar(x, track_building_time, bottom=s, **bar_args)
    ax.text(x, s+track_building_time/2, f"track building ({track_building_time:.2f})", ha="center", va="center")
