import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

def collect_hit_distribution_data(feature_store_dir, train_all_dir, max_events):
    res = {"sim hits": [], "meas hits": [], "reader hits": [], "track hits": [], "target hits": []}
    files = list(feature_store_dir.glob("*-graph.pyg"))

    if len(files) == 0:
        raise RuntimeError(f"No *-graph.pyg files found in {feature_store_dir}")

    for torch_file in tqdm(files[: min(len(files), max_events)]):
        torch_data = torch.load(torch_file, map_location="cpu")
        event_id = torch_file.name[:14]

        raw_hits = pd.read_csv(train_all_dir / f"{event_id}-hits.csv")
        res["sim hits"].append(len(raw_hits))

        meas_idxs = pd.read_csv(
            train_all_dir / f"{event_id}-measurement-simhit-map.csv"
        ).hit_id
        res["meas hits"].append(len(raw_hits.loc[meas_idxs, :]))

        res["reader hits"].append(len(torch_data.r))
        res["track hits"].append(len(np.unique(torch_data.track_edges.flatten())))
        res["target hits"].append(sum((torch_data.pt > 0.5) & (torch_data.nhits >= 3)))

    return res
