import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import uproot
import logging

import acts

from pathlib import Path
from itertools import cycle

import awkward as ak


def main():
    ##############
    # Load stuff #
    ##############

    # Load particles
    particles = ak.to_dataframe(
        uproot.open(snakemake.input[0] + ":particles").arrays()
    ).reset_index(drop=True)
    particles = particles[particles.event_id == 0].copy()

    # Load hits
    hits = uproot.open(snakemake.input[1] + ":hits").arrays(library="pd")
    hits = hits[(hits.event_id == 0) & (hits.tt < 25.0)].copy()
    hits["hit_id"] = np.arange(len(hits))

    hitId_to_particleId = dict(zip(hits.hit_id, hits.particle_id))

    # Load simhit map
    simhit_map = pd.read_csv(snakemake.input[3])
    measId_to_hitID = dict(zip(simhit_map.measurement_id, simhit_map.hit_id))

    # Load measurements
    measurements = pd.read_csv(snakemake.input[2])
    measurements["volume"] = measurements.geometry_id.map(
        lambda geoid: acts.GeometryIdentifier(geoid).volume()
    )
    measurements["hit_id"] = measurements.measurement_id.map(measId_to_hitID)
    measurements["particle_id"] = measurements.hit_id.map(hitId_to_particleId)
    measurements_pixel = measurements[measurements.volume.isin([16, 17, 18])].copy()

    # Load prototracks
    prototracks = pd.read_csv(snakemake.input[4])
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
    print(
        "Prototrack duplicate removal {} -> {}".format(
            prototracks_shape_before_duprem, prototracks.shape
        )
    )

    # Load match df
    matched_df = pd.read_csv(snakemake.input[5])
    matched_df = matched_df[matched_df.event == 0].copy()
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
    poc_match_df = pd.read_csv(snakemake.input[6])
    poc_match_df = poc_match_df[poc_match_df.event == 0].copy()

    # remove unmatched proof of concept particles
    assert len(poc_match_df) == len(matched_df)
    poc_success_pids = poc_match_df[poc_match_df.matched == 1].particle_id.to_numpy()
    matched_df = matched_df[matched_df.particle_id.isin(poc_success_pids)].copy()

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

    ##############################
    # Plot info on all particles #
    ##############################

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    histopts = dict(
        bins=20,
        histtype="barstacked",
        color=["tab:green", "tab:orange", "tab:red"],
        label=["matched", "not matched but found", "lost in GNN"],
        alpha=0.8,
    )

    ax[0].hist(
        [
            matched_df[matched_df.matched == 1].pt,
            matched_df[(matched_df.matched == 0) & (matched_df.nTrackIds > 0)].pt,
            matched_df[(matched_df.matched == 0) & (matched_df.nTrackIds == 0)].pt,
        ],
        range=(0, 5),
        **histopts
    )
    ax[0].legend()
    ax[0].set_xlabel("pT [GeV]")

    ax[1].hist(
        [
            matched_df[matched_df.matched == 1].eta,
            matched_df[(matched_df.matched == 0) & (matched_df.nTrackIds > 0)].eta,
            matched_df[(matched_df.matched == 0) & (matched_df.nTrackIds == 0)].eta,
        ],
        range=(-4, 4),
        **histopts
    )
    ax[1].legend()
    ax[1].set_xlabel("$\eta$")

    fig.suptitle(
        "{}\nDetailed matching histogram".format(snakemake.wildcards[0]),
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(snakemake.output[0])

    # Do same again as efficiency
    def plot_as_eff(ax, df, conditions, key, colors, labels, bins=20, range=None):
        vals, bins = np.histogram(df[key], bins=bins, range=range)
        x = bins[1:]

        base_vals = np.zeros_like(vals)
        for cond, color, label in zip(conditions, colors, labels):
            cond_vals, _ = np.histogram(df[cond][key], bins=bins, range=range)
            new_vals = base_vals + cond_vals

            y0 = np.nan_to_num(base_vals / vals)
            y1 = np.nan_to_num(new_vals / vals)

            ax.fill_between(x, y0, y1, color=color, alpha=0.5, step="pre")
            ax.plot(x, y1, color=color, label=label, drawstyle="steps")

            base_vals = new_vals

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    conditions = [
        matched_df.matched == 1,
        (matched_df.matched == 0) & (matched_df.nTrackIds > 0),
        (matched_df.matched == 0) & (matched_df.nTrackIds == 0),
    ]

    plot_as_eff(
        ax[0],
        matched_df,
        conditions,
        "pt",
        range=(0, 5),
        colors=["tab:green", "tab:orange", "tab:red"],
        labels=["matched", "not matched but found", "lost in GNN"],
    )
    ax[0].set_xlim(0, 5)
    ax[0].legend()
    ax[0].set_xticks([0, 0.5] + list(range(1, 6)))
    ax[0].set_xlabel("pT [GeV]")

    plot_as_eff(
        ax[1],
        matched_df,
        conditions,
        "eta",
        range=(-4, 4),
        colors=["tab:green", "tab:orange", "tab:red"],
        labels=["matched", "not matched but found", "lost in GNN"],
    )
    ax[1].set_xlabel("$\eta$")
    ax[1].legend()

    fig.suptitle(
        "{}\nDetailed matching efficiency".format(snakemake.wildcards[0]),
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(snakemake.output[1])

    ###############################
    # plot info about not matched #
    ###############################

    not_matched = matched_df[matched_df.matched == 0].copy()
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


main()
