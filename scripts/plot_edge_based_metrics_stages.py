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


DEVICE = "cuda"

# Graph
graph = torch.load(snakemake.input[0])
target_mask = (graph.nhits >= snakemake.params["target_min_hits"]) & (graph.pt > snakemake.params["target_min_pt"])

def score_plot(ax, scores, edges):
    ax.hist(scores.detach().cpu().numpy(), bins=40)

    effpur_target  = effpur(graph.track_edges[:,target_mask], edges).values()
    effpur_all = effpur(graph.track_edges, edges).values()

    ax.set_yscale('log')

    ax.text(0.5,0.5, "target eff={:.2f}, pur={:.2f}".format(*effpur_target), transform=plt.gca().transAxes, ha='center')
    ax.text(0.5,0.7, "all eff={:.2f}, pur={:.2f}".format(*effpur_all), transform=plt.gca().transAxes, ha='center')


# Models
embeddingModel = torch.jit.load(snakemake.input[1]).to(DEVICE)
filterModel = torch.jit.load(snakemake.input[2]).to(DEVICE)
gnnModel = torch.jit.load(snakemake.input[3]).to(DEVICE)

# Features
x = torch.stack([
    graph.r/1000.0,
    graph.phi/3.14,
    graph.z/3000.0,
    graph.cell_count,
    graph.cell_val,
    graph.lx,
    graph.ly
]).T.to(torch.float32).to(DEVICE)

# Metric learning
emb = embeddingModel(x.clone()).to(DEVICE)
edge_index = remove_duplicates_with_random_flip(build_edges(emb, emb, r_max=0.2, k_max=100, backend=""))

print("Metric learning:")
print("- all:", effpur(graph.track_edges, edge_index))
print("- target:", effpur(graph.track_edges[:, target_mask], edge_index))

# Filter
filter_scores = torch.sigmoid(filterModel(x[:,:3].clone().float(), edge_index.clone())).detach().cpu()
filter_edges = edge_index[:, filter_scores > 0.5 ]

fig, ax = plt.subplots(1, 2, figsize=(8,5))
score_plot(ax[0], filter_scores, filter_edges)
ax[0].set_title("Filter scores")

# GNN
gnn_scores = torch.sigmoid(gnnModel(x[:,:3].clone(), torch.hstack([filter_edges, filter_edges.flip(0)]))).detach().cpu()
gnn_scores = gnn_scores[:len(gnn_scores)//2]

final_edges = filter_edges[:, gnn_scores > 0.5].detach().cpu().numpy()

score_plot(ax[1], gnn_scores, final_edges)
ax[1].set_title("GNN scores")
fig.tight_layout()
fig.savefig(snakemake.output[0])

# Overview
fig, ax = plt.subplots(1,3, figsize=(12,4))

x_vals = [0,1,2]

ax[0].set_title("Efficiency")
ax[1].set_title("Purity")
ax[2].set_title("Graph size")
ax[2].plot(x_vals, [ e.shape[1] for e in [edge_index, filter_edges, final_edges]], "x-k")

for true_edges, title in zip([graph.track_edges, graph.track_edges[:, target_mask]],
                                ["all edges", "target edges"]):
    effs = []
    purs = []

    for x, stage_edges in enumerate([edge_index, filter_edges, final_edges]):
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
    a.set_xticklabels(["emb", "flt", "gnn"])

ax[1].set_yscale('log')
ax[2].set_yscale('log')

fig.tight_layout()
fig.savefig(snakemake.output[1])
