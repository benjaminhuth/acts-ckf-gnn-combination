import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .detector_plotter import DetectorPlotter


class BatchPlotter:
    def __init__(self, batch, detector_path):
        self.batch = batch

        self.max_z = float(1.1 * max(abs(min(batch.z)), max(batch.z)))
        self.max_r = float(1.1 * max(batch.r))

        self.detector_plotter = DetectorPlotter(
            detector_path, r_max=self.max_r, abs_z_max=self.max_z
        )

    def plot_pids(self, pids, legend=True):
        fig, ax = self.detector_plotter.get_fig_ax(figsize=(15, 7))

        for pid in pids:
            self.plot_pid(pid, ax)

        if legend:
            ax[0].legend()

        ax[0].set_xlim(-self.max_z, self.max_z)
        ax[0].set_ylim(0, self.max_r)
        ax[1].set_xlim(-self.max_r, self.max_r)
        ax[1].set_ylim(-self.max_r, self.max_r)

        return fig, ax

    def plot_parameteric(
            self, pt_min, pt_max, only_pseudo_particles=False, max_tracks=10, legend=True
    ):
        pt_mask = ((self.batch.pt >= pt_min) & (self.batch.pt < pt_max)).numpy()

        if only_pseudo_particles == True and "particle_type" in self.batch:
            pseudo_mask = (self.batch.particle_type == 0).numpy()
        else:
            pseudo_mask = np.ones_like(pt_mask)

        sel_mask = np.logical_and(pt_mask, pseudo_mask)

        sel_pids = np.unique(self.batch.particle_id.numpy()[sel_mask])
        final_pids = np.random.choice(sel_pids, min(len(sel_pids), max_tracks))

        fig, ax = self.plot_pids(final_pids, legend=legend)
        fig.suptitle(
            f"{len(sel_pids)}/{len(np.unique(self.batch.particle_id))} selected particles (plot {len(final_pids)})"
        )

        return fig, ax

    def plot_pid(self, pid, ax):
        mask = (self.batch.particle_id == pid).numpy()
        edges = self.batch.track_edges[:, mask].numpy()

        color = None
        label = "pid: {}".format(pid)

        for a, b in zip(edges[0], edges[1]):
            (line,) = ax[0].plot(
                [self.batch.z[a], self.batch.z[b]],
                [self.batch.r[a], self.batch.r[b]],
                c=color,
                label=label,
            )
            color = line._color
            label = None

            ax[1].plot(
                [self.batch.x[a], self.batch.x[b]],
                [self.batch.y[a], self.batch.y[b]],
                c=color,
            )[0]._color

        # if not self.truth_hits is None and true_hits_if_possible:
        # thits = self.truth_hits[ truth_hits.particle_id == pid ].sort_values("index").copy()

        # thits["tpt"] = np.hypot(thits.tpx, thits.tpy)
        # thits["tp"] = np.hypot(thits.tpt, thits.tpz)

        # thits.tpx /= thits.tpt
        # thits.tpy /= thits.tpt

        # thits.tpz /= thits.tp
        # thits.tpt /= thits.tp

        # ax[0].plot(thits.tz, np.hypot(thits.tx, thits.ty), ":", color=color)
        # ax[0].quiver(thits.tz, np.hypot(thits.tx, thits.ty), thits.tpz, thits.tpt, color=line._color)
        # ax[1].plot(thits.tx, thits.ty, ":", color=line._color)
        # ax[1].quiver(thits.tx, thits.ty, thits.tpx, thits.tpy, color=line._color)
