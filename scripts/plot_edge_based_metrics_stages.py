import sys
import yaml
import os
from pathlib import Path

import torch
import torch_geometric
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.sparse as sps
import tqdm

sys.path.append("/home/iwsatlas1/bhuth/exatrkx/gnn4itk/commonframework")
from gnn4itk_cf.stages.data_reading import ActsReader
from gnn4itk_cf.stages.graph_construction.models.utils import build_edges
from gnn4itk_cf.stages.edge_classifier.models.filter import Filter

DEVICE = "cpu" if os.environ["CUDA_VISIBLE_DEVICES"] == "" else "cuda"

def cantor_pairing(a):
    a = np.sort(a, axis=0)
    return a[1] + ((a[0] + a[1]) * (a[0] + a[1] + 1)) // 2

def cantor_pairing_inv(z):
    def f(w):
        return (w * (w + 1)) // 2

    def q(z):
        return np.floor(0.5 * (np.sqrt(8 * z + 1) - 1))

    res = np.zeros((2, len(z)), dtype=int)
    res[1] = z - f(q(z))
    res[0] = q(z) - res[1]

    return res

def effpur(true, pred):
    assert true.shape[0] == 2
    assert pred.shape[0] == 2
    assert true.shape[1] > 1
    assert pred.shape[1] > 1

    try:
        true = np.sort(true.detach().cpu().numpy(), axis=0)
    except:
        pass

    try:
        pred = np.sort(pred.detach().cpu().numpy(), axis=0)
    except:
        pass

    cantor_true = cantor_pairing(true)
    cantor_pred = cantor_pairing(pred)
    cantor_intersection = np.intersect1d(cantor_true, cantor_pred)

    return {
        "eff": len(cantor_intersection)
        / len(cantor_true),  # if len(cantor_true) > 0 else 0,
        "pur": len(cantor_intersection)
        / len(cantor_pred),  # if len(cantor_pred) > 0 else 0,
    }

def remove_duplicates_with_random_flip(edge_index): # From MetricLearing Graph construction module
    edge_index[:, edge_index[0] > edge_index[1]] = edge_index[:, edge_index[0] > edge_index[1]].flip(0)
    edge_index, edge_inverse = edge_index.unique(return_inverse=True, dim=-1)
    #y = torch.zeros_like(edge_index[0], dtype=y.dtype).scatter(0, edge_inverse, y)
    #truth_map[truth_map >= 0] = edge_inverse[truth_map[truth_map >= 0]]
    #truth_map = truth_map[:track_edges.shape[1]]

    random_flip = torch.randint(2, (edge_index.shape[1],), dtype=torch.bool)
    edge_index[:, random_flip] = edge_index[:, random_flip].flip(0)

    return edge_index



# Workaround because we must call this from shell
class Snakemake:
    input = os.environ["SNAKEMAKE_INPUT"].split()
    output = os.environ["SNAKEMAKE_OUTPUT"].split()
    
target_min_hits = int(os.environ["TARGET_MIN_HITS"])
target_min_pt = float(os.environ["TARGET_MIN_PT"])
cuts = [ float(c) for c in os.environ["CUTS"].split() ]


snakemake = Snakemake()

# Graph
graph = torch.load(snakemake.input[0])
target_mask = (graph.nhits >= target_min_hits) & (graph.pt > target_min_pt)

def score_plot(ax, scores, edges):
    scores = scores.detach().cpu().numpy()

    v, bins = np.histogram(scores, bins=40)
    v = v / max(v)

    ax.bar(bins[:-1], v, width=np.diff(bins), color="lightgrey")

    eff_all_list = []
    pur_all_list = []
    eff_tgt_list = []
    pur_tgt_list = []

    test_scores = np.linspace(0.0,0.99,50)

    # print("score\ttarget eff\ttarget pur")
    for score in test_scores:
        score_edges = edges[ :, scores > score ]

        eff_tgt, pur_tgt = effpur(graph.track_edges[:,target_mask], score_edges).values()
        eff_tgt_list.append(eff_tgt)
        pur_tgt_list.append(pur_tgt)

        eff_all, pur_all = effpur(graph.track_edges, score_edges).values()
        eff_all_list.append(eff_all)
        pur_all_list.append(pur_all)

        # print(f"{score:.2f}\t{eff_tgt:.3f}\t{pur_tgt:.3f}")

    ax.plot(test_scores, eff_all_list, label="eff all", color="tab:blue")
    ax.plot(test_scores, pur_all_list, label="pur all", color="tab:blue", ls=":")
    ax.plot(test_scores, eff_tgt_list, label="eff target", color="tab:orange")
    ax.plot(test_scores, pur_tgt_list, label="pur target", color="tab:orange", ls=":")

    ax.vlines(0.5, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], ls=":", color="black")

    #ax.set_yscale('log')

    ax.legend(loc='lower right')
    #ax.text(0.5,0.5, "target eff={:.2f}, pur={:.2f}".format(*effpur_target), transform=plt.gca().transAxes, ha='center')
    #ax.text(0.5,0.7, "all eff={:.2f}, pur={:.2f}".format(*effpur_all), transform=plt.gca().transAxes, ha='center')


# Models
embeddingModel = torch.jit.load(snakemake.input[1])
filterModel = torch.jit.load(snakemake.input[2])
gnnModel = torch.jit.load(snakemake.input[3])

classifier_models = {
    "filter": filterModel,
    "GNN": gnnModel
}

# TODO make this visible to snakemake
gnnModel2Path = Path(snakemake.input[3]).parent / "gnn2.pt"
if gnnModel2Path.exists():
    print("GNN2 found, use it!")
    classifier_models["GNN2"] = torch.jit.load(gnnModel2Path)

# Features
x = torch.stack([
    graph.r/1000.0,
    graph.phi/3.14,
    graph.z/3000.0,
    graph.cell_count,
    graph.cell_val,
    graph.lx,
    graph.ly
]).T.to(torch.float32)

# Metric learning
print("Metric learning")

with torch.inference_mode():
    emb = embeddingModel.to(DEVICE)(x.to(DEVICE))
    edge_index = remove_duplicates_with_random_flip(build_edges(emb, emb, r_max=0.2, k_max=100, backend="")).detach().cpu()
    del emb

print("- all:", effpur(graph.track_edges, edge_index))
print("- target:", effpur(graph.track_edges[:, target_mask], edge_index))

def classify(model, e):
    with torch.inference_mode():
        return torch.sigmoid(model.to(DEVICE)(x[:,:3].float().to(DEVICE), e.to(DEVICE))).detach().cpu()

# Loop over classifier stages
stage_edge_list = [edge_index]

fig, ax = plt.subplots(1, len(classifier_models), figsize=(8,5))

for (stage_name, stage_model), ax in zip(classifier_models.items(), ax):
    print(stage_name)

    if "GNN" in stage_name:
        input_edges = torch.hstack([ stage_edge_list[-1], stage_edge_list[-1].flip(0) ])
    else:
        input_edges = stage_edge_list[-1]

    scores = classify(stage_model, input_edges)

    if "GNN" in stage_name:
        scores = scores[:len(scores)//2]

    stage_edge_list.append(stage_edge_list[-1][:, scores > cuts[0] ])

    print("- all:", effpur(graph.track_edges, stage_edge_list[-1]))
    print("- target:", effpur(graph.track_edges[:, target_mask], stage_edge_list[-1]))

    assert len(scores) == stage_edge_list[-2].shape[1]
    score_plot(ax, scores, stage_edge_list[-2])
    ax.set_title(f"{stage_name} scores")


fig.tight_layout()
fig.savefig(snakemake.output[0])

# Overview plot
print("Final plot")
fig, ax = plt.subplots(1,3, figsize=(12,4))

x_vals = np.arange(len(stage_edge_list))

ax[0].set_title("Efficiency")
ax[1].set_title("Purity")
ax[2].set_title("Graph size")
ax[2].plot(x_vals, [ e.shape[1] for e in stage_edge_list], "x-k")

for true_edges, title in zip([graph.track_edges, graph.track_edges[:, target_mask]],
                                ["all edges", "target edges"]):
    effs = []
    purs = []

    for x, stage_edges in enumerate(stage_edge_list):
        eff, pur = effpur(true_edges, stage_edges).values()
        effs.append(eff)
        purs.append(pur)

        ax[0].text(x, eff, f"{eff:.2f}", ha="center", color="dimgrey")
        ax[1].text(x, pur, f"{pur:.2f}", ha="center", color="dimgrey")

    ax[0].plot(x_vals, effs, "x-", label=title)
    ax[1].plot(x_vals, purs, "x-", label=title)

for a in ax:
    a.legend()
    a.set_xticks(x_vals)
    a.set_xticklabels(["emb",] + list(classifier_models.keys()))

ax[1].set_yscale('log')
ax[2].set_yscale('log')

fig.tight_layout()
fig.savefig(snakemake.output[1])
