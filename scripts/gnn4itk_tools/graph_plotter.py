import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from .detector_plotter import DetectorPlotter

class GraphPlotter:
    def __init__(self, graph, r_max, abs_z_max, detector_file="/home/iwsatlas1/bhuth/exatrkx/data/2K_geant4_pixel_geodigi/detectors.csv", **kwargs):
        plotter = DetectorPlotter(detector_file, r_max, abs_z_max)

        self.graph = graph.cpu()
        self.fig, self.ax = plotter.get_fig_ax(**kwargs)

        self.ranges = {
            "x": [200, -200],
            "y": [200, -200],
            "z": [1000, -1000],
            "r": [200, 0],
        }

    def plot_edges(self, edges, **kwargs):
        # array of shape [lines, line-length (=2), 2]
        # [
        #   [ [x0,y0], [x1,y1] ],
        #   [ [x0,y0], [x1,y1] ],
        #   ...
        # ]

        def make_segs(ca, cb):
            a = self.graph[ca].numpy()[edges].T
            b = self.graph[cb].numpy()[edges].T

            return np.transpose(np.hstack([a, b]).reshape(-1,2,2), (0,2,1))

        segs_zr = make_segs('z', 'r')
        lines_zr = LineCollection(segs_zr, **kwargs)
        self.ax[0].add_collection(lines_zr)

        segs_xy = make_segs('x', 'y')
        lines_xy = LineCollection(segs_xy, **kwargs)
        self.ax[1].add_collection(lines_xy)

        for c in ["x", "y", "z", "r"]:
            cf = self.graph[c].numpy()[edges].flatten()
            self.ranges[c][0] = min(self.ranges[c][0], min(cf))
            self.ranges[c][1] = max(self.ranges[c][1], max(cf))

    def adjust_limits(self):
        def increase_range(r):
            assert r[1] >= r[0]
            d = r[1] - r[0]
            return r[0] - 0.1*d, r[1] + 0.1*d

        self.ax[0].set_xlim(*increase_range(self.ranges["z"]))
        self.ax[0].set_ylim(*increase_range(self.ranges["r"]))
        self.ax[1].set_xlim(*increase_range(self.ranges["x"]))
        self.ax[1].set_ylim(*increase_range(self.ranges["y"]))
