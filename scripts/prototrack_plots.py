import logging
from pathlib import Path
from itertools import cycle
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib
import numpy as np
import awkward as ak
import uproot

import tqdm
from tqdm.contrib.concurrent import process_map

import acts




##############
# Load stuff #
##############

def read_event(event_nr, particles, hits, gnn_ckf_matching_df, poc_matching_df):
    null_str = "000000000"
    event_str = f"{event_nr:09}"

    # Prepare particles
    particles = particles[particles.event_id == event_nr].copy()

    # Prepare hits
    hits = hits[(hits.event_id == event_nr) & (hits.tt < 25.0)].copy()
    hits["hit_id"] = np.arange(len(hits))

    hitId_to_particleId = dict(zip(hits.hit_id, hits.particle_id))

    # Load simhit map
    simhit_map = pd.read_csv(snakemake.input[3].replace(null_str, event_str))
    measId_to_hitID = dict(zip(simhit_map.measurement_id, simhit_map.hit_id))

    # Load measurements
    measurements = pd.read_csv(snakemake.input[2].replace(null_str, event_str))
    measurements["volume"] = measurements.geometry_id.map(
        lambda geoid: acts.GeometryIdentifier(geoid).volume()
    )
    measurements["hit_id"] = measurements.measurement_id.map(measId_to_hitID)
    measurements["particle_id"] = measurements.hit_id.map(hitId_to_particleId)
    measurements_pixel = measurements[measurements.volume.isin([16, 17, 18])].copy()

    # Load prototracks
    prototracks = pd.read_csv(snakemake.input[4].replace(null_str, event_str))
    prototracks_shape_before_duprem = prototracks.shape
    prototracks = prototracks[
        ~prototracks.duplicated(["x", "y", "z", "measurementId"])
    ].copy()
    prototracks["hit_id"] = prototracks["measurementId"].map(measId_to_hitID)
    prototracks["tx"] = prototracks.hit_id.map(dict(zip(hits.hit_id, hits.tx)))
    prototracks["ty"] = prototracks.hit_id.map(dict(zip(hits.hit_id, hits.ty)))
    prototracks["tz"] = prototracks.hit_id.map(dict(zip(hits.hit_id, hits.tz)))
    prototracks["geometry_id"] = prototracks.hit_id.map(
        dict(zip(hits.hit_id, hits.geometry_id))
    )
    prototracks["particle_id"] = prototracks.hit_id.map(hitId_to_particleId)

    # There should not be duplicates here
    assert prototracks_shape_before_duprem[0] == prototracks.shape[0]

    # Load match df
    # TODO This is currently done very inefficiently...
    matched_df = gnn_ckf_matching_df[gnn_ckf_matching_df.event == event_nr].copy()

    matched_df["eta"] = matched_df.particle_id.map(
        dict(zip(particles.particle_id, particles.eta))
    )
    matched_df["pt"] = matched_df.particle_id.map(
        dict(zip(particles.particle_id, particles.pt))
    )
    matched_df["pix_nhits"] = matched_df.particle_id.map(
        dict(
            measurements[measurements.volume.isin([16, 17, 18])]
            .groupby("particle_id")
            .count()
            .hit_id
        )
    )

    # Load match df of proof of concept run
    # TODO This is currently done very inefficiently...
    poc_matching_df = poc_matching_df[poc_matching_df.event == event_nr].copy()

    # flag unmatched proof of concept particles
    assert len(poc_matching_df) == len(matched_df)
    poc_success_pids = poc_matching_df[poc_matching_df.matched == 1].particle_id.to_numpy()
    matched_df["matched_proof_of_concept"] = matched_df.particle_id.isin(poc_success_pids)

    ###############################################
    # Combine matching and prototrack information #
    ###############################################
    df = pd.DataFrame()

    for pid in matched_df.particle_id.to_numpy():
        info = {}

        info["particle_id"] = pid

        trkMeasurements = measurements_pixel[measurements_pixel.particle_id == pid]
        info["nMeasPixel"] = len(trkMeasurements)

        trackIds = prototracks[prototracks.particle_id == pid].trackId.to_numpy()
        info["nTrackIds"] = len(trackIds)

        trackIds, count = np.unique(trackIds, return_counts=True)
        info["nUniqueTrackIds"] = len(trackIds)

        if len(trackIds) > 0:
            info["majTrackId"] = trackIds[np.argmax(count)]
            info["foundMajTrackIds"] = max(count)
            info["totalMajTrackIds"] = len(
                prototracks[prototracks.trackId == info["majTrackId"]]
            )

        for k in info.keys():
            info[k] = [info[k]]

        df = pd.concat([df, pd.DataFrame(info)])

    df = df.reset_index(drop=True)

    matched_df = pd.merge(matched_df, df, on="particle_id")

    matched_df["pixel_track_eff"] = matched_df.foundMajTrackIds / matched_df.nMeasPixel
    matched_df["pixel_track_pur"] = (
        matched_df.foundMajTrackIds / matched_df.totalMajTrackIds
    )

    matched_df["eta"] = matched_df.particle_id.map(
        dict(zip(particles.particle_id, particles.eta))
    )
    matched_df["pt"] = matched_df.particle_id.map(
        dict(zip(particles.particle_id, particles.pt))
    )
    matched_df["pix_nhits"] = matched_df.particle_id.map(
        dict(measurements_pixel.groupby("particle_id").count().hit_id)
    )

    return matched_df

# Load particles
particles = ak.to_dataframe(
    uproot.open(snakemake.input[0] + ":particles").arrays()
).reset_index(drop=True)

# Load hits
hits = uproot.open(snakemake.input[1] + ":hits").arrays(library="pd")

# Load matching dfs
gnn_ckf_matching_df = pd.read_csv(snakemake.input[5])
poc_matching_df = pd.read_csv(snakemake.input[6])

# Load events
matched_df = pd.DataFrame()

for event_nr in tqdm.tqdm(range(max(hits.event_id)+1), desc="Read events..."):
    try:
        matched_df = pd.concat([matched_df, read_event(event_nr, particles, hits, gnn_ckf_matching_df, poc_matching_df)])
    except Exception as e:
        print("ERROR reading event", event_nr)
        print(e)

assert len(matched_df) > 0

##############################
# Plot info on all particles #
##############################

# Efficiency histogram
def plot_as_eff(ax, df, conditions, key, color, label, bins, range, alpha, **kwargs):
    vals, bins = np.histogram(df[key], bins=bins, range=range)
    x = bins[1:]

    base_vals = np.zeros_like(vals)
    for cond, c, l in zip(conditions, color, label):
        cond_vals, _ = np.histogram(df[cond][key], bins=bins, range=range)
        new_vals = base_vals + cond_vals

        y0 = np.nan_to_num(base_vals / vals)
        y1 = np.nan_to_num(new_vals / vals)

        ax.fill_between(x, y0, y1, color=c, alpha=alpha, step="pre")
        ax.plot(x, y1, color=c, label=l, drawstyle="steps", lw=1)

        base_vals = new_vals

# Default histopts
common_opts = dict(
    histtype="barstacked",
    color=["tab:green", "tab:orange", "tab:red", "grey"],
    label=["matched", "matched in GNN, not matched in CKF", "not matched in GNN", "not mached in proof-of-concept"],
    alpha=0.5,
)

pt_opts = dict(
    bins=np.logspace(0,2,20),
    range=(1,100),
)

eta_opts = dict(
    bins=20,
    range=(-3,3),
)

# Define conditions
matched = (matched_df.matched == 1)
has_tracks = (matched_df.nTrackIds > 0)
poc_matched = (matched_df.matched_proof_of_concept == 1)

conditions = [
    matched & poc_matched,
    ~matched & has_tracks & poc_matched,
    ~matched & ~has_tracks & poc_matched,
    ~poc_matched,
]

# Plot as standard histograms
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[1].hist([ matched_df[c].pt for c in conditions ], **pt_opts, **common_opts)
ax[1].set_xscale('log')
ax[1].set_xlabel("pT [GeV]")
ax[1].set_xticks([1,3,10,30,100])
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax[1].legend()

ax[0].hist([ matched_df[c].eta for c in conditions ], **eta_opts, **common_opts)
ax[0].set_xlabel("$\eta$")
ax[0].set_xticks(np.arange(-3,4))
ax[0].legend()

# fig.suptitle(
#     "{}\nDetailed matching histogram".format(snakemake.wildcards[0]),
#     fontweight="bold",
# )
fig.tight_layout()
fig.savefig(snakemake.output[0])

# Plot as efficiency histograms
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

plot_as_eff(ax[1], matched_df, conditions, "pt", **pt_opts, **common_opts)
# ax[1].legend()
ax[1].set_xscale('log')
ax[1].set_xlabel("pT [GeV]")
ax[1].set_title("Detailed matching efficiency vs $p_T$")
ax[1].set_xticks([1,3,10,30,100])
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plot_as_eff(ax[0], matched_df, conditions, "eta", **eta_opts, **common_opts)
ax[0].set_xlabel("$\eta$")
ax[0].set_title("Detailed matching efficiency vs $\eta$")
ax[0].set_xticks(np.arange(-3,4))
ax[0].legend()

# fig.suptitle(
#     "{}\nDetailed matching efficiency".format(snakemake.wildcards[0]),
#     fontweight="bold",
# )
fig.tight_layout()
fig.savefig(snakemake.output[1])

###############################
# plot info about not matched #
###############################

# Show plots for not matched particles that have been matched in the proof of concept
not_matched = matched_df[(matched_df.matched == 0) & (matched_df.matched_proof_of_concept == 1)].copy()
fig, ax = plt.subplots(2, 3, figsize=(10, 6))
ax = ax.flatten()

ax[0].bar(*np.unique(not_matched.nMeasPixel, return_counts=True))
ax[0].set_title("# total hits in pixel")
ax[0].set_ylabel("particle count")
ax[0].set_xticks(np.arange(10))

ax[1].bar(*np.unique(not_matched.nTrackIds, return_counts=True))
ax[1].set_title("# found hits in pixel\n(in all prototracks)")
ax[1].set_ylabel("particle count")
ax[1].set_xticks(np.arange(min(not_matched.nTrackIds), 10))

ax[2].bar(*np.unique(not_matched.foundMajTrackIds, return_counts=True))
ax[2].set_title("# found hits in pixel\n(in maj prototrack)")
ax[2].set_ylabel("particle count")
ax[2].set_xticks(np.arange(10))

ax[3].bar(*np.unique(not_matched.nUniqueTrackIds, return_counts=True))
ax[3].set_title("prototracks per particle")
ax[3].set_ylabel("particle count")
ax[3].set_xticks(
    np.arange(
        min(not_matched.nUniqueTrackIds), max(not_matched.nUniqueTrackIds) + 1
    )
)

nFinite = sum(np.isfinite(not_matched.pixel_track_pur))
ax[4].hist(not_matched.pixel_track_pur, bins=20)
ax[4].set_title("maj track pur in pixel")
ax[4].text(
    0.1,
    0.7,
    "applicable to\n{} particles\n({:.2%})".format(
        nFinite, nFinite / len(not_matched)
    ),
    transform=ax[4].transAxes,
)
ax[4].set_ylabel("particle count")
ax[4].set_xlim(0, 1)

nFinite = sum(np.isfinite(not_matched.pixel_track_eff))
ax[5].hist(not_matched.pixel_track_eff, bins=20)
ax[5].set_title("maj track eff in pixel")
ax[5].text(
    0.1,
    0.7,
    "applicable to\n{} particles\n({:.2%})".format(
        nFinite, nFinite / len(not_matched)
    ),
    transform=ax[5].transAxes,
)
ax[5].set_ylabel("particle count")
ax[5].set_xlim(0, 1)

fig.suptitle(
    "{}\n"
    "Statistics for particles NOT matched in CKF performance writer "
    "BUT matched in proof of concept \n({}/{}, {:.2%})".format(
        str(snakemake.wildcards[0]),
        len(not_matched),
        len(matched_df),
        len(not_matched) / len(matched_df),
    ),
    fontweight="bold",
)

fig.tight_layout()
fig.savefig(snakemake.output[2])

# plt.show()
