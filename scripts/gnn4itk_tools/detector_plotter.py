import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DetectorPlotter:
    def __init__(self, detector_path, r_max=np.inf, abs_z_max=np.inf):
        self.detector = pd.read_csv(detector_path)
        self.detector["cr"] = np.hypot(self.detector.cx, self.detector.cy)
        self.detector = self.detector[abs(self.detector.cz) < abs_z_max].copy()
        self.detector = self.detector[self.detector.cr < r_max].copy()

    def get_fig_ax(self, figsize=None, s=5, color="lightgrey", alpha=1.0, ax_config=(1,2)):
        assert ax_config[0] * ax_config[1] == 2
        fig, ax = plt.subplots(*ax_config, figsize=figsize)

        # Draw pixel a bit nicer
        for r in [20.0, 70.0, 111.0, 170.0]:
            ax[0].plot([-500, 500], [r, r], color=color, zorder=-20)
            ax[1].add_patch(plt.Circle((0,0), radius=r, ec=color, fill=None, zorder=-20))

        for z in [600.0, 700.0, 850.0, 950.0, 1100.0, 1300.0]:
            for s in [-1, 1]:
                ax[0].plot([s*z, s*z], [40, 175], color=color, zorder=-20)


        ax[0].scatter(self.detector.cz, self.detector.cr, s=s, color=color, alpha=1.0, zorder=-20)
        ax[0].set_xlabel("z [mm]")
        ax[0].set_ylabel("r [mm]")

        ax[1].scatter(self.detector.cx, self.detector.cy, s=s, color=color, alpha=1.0, zorder=-20)
        ax[1].set_xlabel("x [mm]")
        ax[1].set_ylabel("y [mm]")

        return fig, ax
