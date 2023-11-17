import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

graph = torch.load(snakemake.input[0])
print(graph)

embeddingModel = torch.jit.load(snakemake.input[1])

DEVICE = "cpu" if os.environ["CUDA_VISIBLE_DEVICES"] == "" else "cuda"

# Do the inference
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

with torch.inference_mode():
    emb = embeddingModel.to(DEVICE)(x.to(DEVICE)).detach().cpu().numpy()


# Establish mapping between pid and hit index
pids = np.concatenate([ graph.particle_id, graph.particle_id ])
hits = np.concatenate([ graph.track_edges[0], graph.track_edges[1] ])

hits, idxs = np.unique(hits, return_index=True)
pids = pids[idxs]

# Select some pids to plot
pids_to_plot = np.unique(pids)[:50]
mask = np.isin(pids, pids_to_plot)

sel_pids = pids[mask]
sel_hits = hits[mask]

sel_emb = emb[sel_hits]

# Plot TSNE reduction of the space
print("Make TSNE reduction")
tsne = TSNE(n_iter=2000, n_jobs=-1, verbose=2, random_state=239874)
emb_transformed = tsne.fit_transform(sel_emb)

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].scatter(graph.x[sel_hits], graph.y[sel_hits], s=3, color="grey")
ax[1].scatter(emb_transformed[:,0], emb_transformed[:,1], s=3, color="grey")

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

np.random.seed(9765)
np.random.shuffle(pids_to_plot)

for pid, c in zip(pids_to_plot[:4], colors):
    track_hits_sel = (sel_pids == pid)
    track_hits_all = sel_hits[track_hits_sel]

    ax[0].scatter(graph.x[track_hits_all], graph.y[track_hits_all], s=8, color=c)
    ax[1].scatter(emb_transformed[track_hits_sel,0], emb_transformed[track_hits_sel,1], s=8, color=c)

ax[0].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
ax[0].set_title("Real space")

ax[1].set_xlabel("$emb_0$")
ax[1].set_ylabel("$emb_1$")
ax[1].set_title("TSNE reduced embedding")

fig.tight_layout()
fig.savefig(snakemake.output[0])
