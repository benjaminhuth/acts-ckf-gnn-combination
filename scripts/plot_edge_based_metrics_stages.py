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
from gnn4itk_cf.stages.graph_construction.models.utils import graph_intersection


plt.rcParams.update({"font.size": 16})

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


def remove_duplicates_with_random_flip(
    edge_index,
):  # From MetricLearing Graph construction module
    edge_index[:, edge_index[0] > edge_index[1]] = edge_index[
        :, edge_index[0] > edge_index[1]
    ].flip(0)
    edge_index, edge_inverse = edge_index.unique(return_inverse=True, dim=-1)
    # y = torch.zeros_like(edge_index[0], dtype=y.dtype).scatter(0, edge_inverse, y)
    # truth_map[truth_map >= 0] = edge_inverse[truth_map[truth_map >= 0]]
    # truth_map = truth_map[:track_edges.shape[1]]

    random_flip = torch.randint(2, (edge_index.shape[1],), dtype=torch.bool)
    edge_index[:, random_flip] = edge_index[:, random_flip].flip(0)

    return edge_index


# Workaround because we must call this from shell
class Snakemake:
    input = os.environ["SNAKEMAKE_INPUT"].split()
    output = os.environ["SNAKEMAKE_OUTPUT"].split()


target_min_hits = int(os.environ["TARGET_MIN_HITS"])
target_min_pt = float(os.environ["TARGET_MIN_PT"])
cuts = [float(c) for c in os.environ["CUTS"].split()]


snakemake = Snakemake()

# Graph
graph = torch.load(snakemake.input[0])


print(graph)

target_mask = (graph.nhits >= target_min_hits) & (graph.pt > target_min_pt)


def score_plot(ax, scores, edges, cut_to_draw):
    scores = scores.detach().cpu().numpy()

    v, bins = np.histogram(scores, bins=100)
    v = v / max(v)

    ax.bar(bins[:-1], v, width=np.diff(bins), color="lightgrey")

    eff_all_list = []
    pur_all_list = []
    eff_tgt_list = []
    pur_tgt_list = []

    test_scores = np.linspace(0.0, 0.99, 50)

    # print("score\ttarget eff\ttarget pur")
    for score in test_scores:
        score_edges = edges[:, scores > score]

        eff_tgt, pur_tgt = effpur(
            graph.track_edges[:, target_mask], score_edges
        ).values()
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

    ax.vlines(
        cut_to_draw,
        ymin=ax.get_ylim()[0],
        ymax=ax.get_ylim()[1],
        ls=":",
        lw=3,
        color="black",
    )

    # ax.set_yscale('log')

    # ax.legend(loc='lower right')
    # ax.text(0.5,0.5, "target eff={:.2f}, pur={:.2f}".format(*effpur_target), transform=plt.gca().transAxes, ha='center')
    # ax.text(0.5,0.7, "all eff={:.2f}, pur={:.2f}".format(*effpur_all), transform=plt.gca().transAxes, ha='center')


def score_plot_pos_neg(ax, scores, edges):
    # _, y, truth_map = graph_intersection(
    #     edges,
    #     graph.track_edges,
    #     return_y_pred=True,
    #     return_truth_to_pred=True,
    #     unique_pred=False,
    # )
    # y = y.numpy()

    cantor_edges = cantor_pairing(edges)
    cantor_all = cantor_pairing(graph.track_edges)
    cantor_target = cantor_pairing(graph.track_edges[:, target_mask])

    y_all = np.isin(cantor_edges, cantor_all)
    y_target = np.isin(cantor_edges, cantor_target)

    # target_map = truth_map[target_mask].numpy()
    # target_map = target_map[ target_map != -1 ]
    #
    # target_index_mask = np.zeros(len(y), dtype=bool)
    # target_index_mask[ target_map ] = True

    pos_edges = scores.numpy()[y_all & ~y_target]
    pos_target_edges = scores.numpy()[y_target]
    neg_edges = scores.numpy()[~y_all]

    # print(pos_edges)
    # print(neg_edges)

    # ax.hist([neg_edges, pos_edges], bins=100, histtype="bar", stacked=True, color=["tab:red", "tab:green"])
    ax.hist(
        [pos_target_edges, pos_edges, neg_edges],
        bins=50,
        histtype="bar",
        stacked=True,
        color=["darkgreen", "forestgreen", "tab:red"],
        label=["true (target)", "true (non-target)", "false"],
    )

    # ax.hist([pos_edges], bins=100, histtype="bar", stacked=True, color=["tab:green"])

    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("score")


def eta_eff_plot(ax, edges):
    cantor_edges = cantor_pairing(edges)
    cantor_true = cantor_pairing(graph.track_edges[:, target_mask])
    true_positive = edges[:, np.isin(cantor_edges, cantor_true)]

    eta_true = graph.eta[np.sort(graph.track_edges[:, target_mask], 0)[0]]
    eta_true_positive = graph.eta[np.sort(true_positive, 0)[0]]

    x1, bins = np.histogram(eta_true, bins=20, range=(-3.5, 3.5))
    x2, _ = np.histogram(eta_true_positive, bins=bins)

    assert (x1 > 0).all()
    assert (x2 <= x1).all()

    eff = x2 / x1

    ax.bar(bins[:-1], eff, np.diff(bins))
    ax.set_ylabel("edge efficiency")
    ax.set_xlabel("$\eta$")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.linspace(-3.0, 3.0, 7))


# Models
embeddingModel = torch.jit.load(snakemake.input[1])
filterModel = torch.jit.load(snakemake.input[2])
gnnModel = torch.jit.load(snakemake.input[3])

classifier_models = {"Filter": filterModel, "GNN": gnnModel}

# TODO make this visible to snakemake
gnnModel2Path = Path(snakemake.input[3]).parent / "gnn2.pt"
if gnnModel2Path.exists():
    print("GNN2 found, use it!")
    classifier_models["GNN2"] = torch.jit.load(gnnModel2Path)

# Features
x = torch.stack(
    [
        graph.r / 1000.0,
        graph.phi / 3.14,
        graph.z / 3000.0,
        graph.cell_count,
        graph.cell_val,
        graph.lx,
        graph.ly,
    ]
).T.to(torch.float32)

# Metric learning
print("Metric learning")

with torch.inference_mode():
    emb = embeddingModel.to(DEVICE)(x.to(DEVICE))
    edge_index = (
        remove_duplicates_with_random_flip(
            build_edges(emb, emb, r_max=0.2, k_max=100, backend="")
        )
        .detach()
        .cpu()
    )
    del emb

print("- all:", effpur(graph.track_edges, edge_index))
print("- target:", effpur(graph.track_edges[:, target_mask], edge_index))


def classify(model, e):
    with torch.inference_mode():
        return (
            torch.sigmoid(model.to(DEVICE)(x[:, :3].float().to(DEVICE), e.to(DEVICE)))
            .detach()
            .cpu()
        )


# Loop over classifier stages
stage_edge_list = [edge_index]

fig, axes = plt.subplots(
    1, len(classifier_models), figsize=(6 * len(classifier_models), 5)
)

fig2, axes2 = plt.subplots(
    1, len(classifier_models), figsize=(6 * len(classifier_models), 5)
)

fig3, axes3 = plt.subplots(
    1, len(classifier_models), figsize=(6 * len(classifier_models), 5)
)

for (stage_name, stage_model), ax, ax2, ax3, cut in zip(
    classifier_models.items(), axes, axes2, axes3, cuts
):
    print(stage_name)

    if "GNN" in stage_name:
        input_edges = torch.hstack([stage_edge_list[-1], stage_edge_list[-1].flip(0)])
    else:
        input_edges = stage_edge_list[-1]

    scores = classify(stage_model, input_edges)

    if "GNN" in stage_name:
        scores = scores[: len(scores) // 2]

    stage_edge_list.append(stage_edge_list[-1][:, scores > cut])

    print("- all:", effpur(graph.track_edges, stage_edge_list[-1]))
    print("- target:", effpur(graph.track_edges[:, target_mask], stage_edge_list[-1]))

    assert len(scores) == stage_edge_list[-2].shape[1]
    score_plot(ax, scores, stage_edge_list[-2], cut)
    ax.set_title(f"{stage_name} scores")

    score_plot_pos_neg(ax2, scores, stage_edge_list[-2])
    ax2.set_title(f"{stage_name} scores")

    eta_eff_plot(ax3, stage_edge_list[-1])
    ax3.set_title(f"{stage_name} scores")


axes[-1].legend(bbox_to_anchor=(1.1, 0.5))

fig.tight_layout()
fig.savefig(snakemake.output[0])

fig2.tight_layout()
fig2.savefig(snakemake.output[2])

fig3.tight_layout()
fig3.savefig(snakemake.output[3])

plt.show()

# Overview plot
print("Final plot")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

x_vals = np.arange(len(stage_edge_list))

ax[0].set_title("Efficiency")
ax[1].set_title("Purity")
ax[2].set_title("Graph size")
ax[2].plot(x_vals, [e.shape[1] for e in stage_edge_list], "x-k")

for true_edges, title in zip(
    [graph.track_edges, graph.track_edges[:, target_mask]],
    ["all edges", "target edges"],
):
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
    a.set_xticklabels(
        [
            "Metric\nLearning",
        ]
        + list(classifier_models.keys())
    )

ax[1].set_yscale("log")
ax[2].set_yscale("log")

ax[2].set_ylim(0.7 * 1e4, 1.3 * 1e6)
ax[2].set_yticks([1e4, 1e5, 1e6])

fig.tight_layout()
fig.savefig(snakemake.output[1])
