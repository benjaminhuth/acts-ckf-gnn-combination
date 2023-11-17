from itertools import cycle

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import awkward as ak
import uproot
import tqdm

import acts

from gnn4itk_tools.detector_plotter import DetectorPlotter


class PrototrackPlotter(DetectorPlotter):
    def __init__(self, graph, particles):
        super().__init__(snakemake.input[0], r_max=200, abs_z_max=1500)
        self.graph = graph
        self.particles = particles

    def plot_prototrack(
        self,
        prototrack: pd.DataFrame,
        random_factor=0.0,
        color_by_particles=True,
        large=False,
        text=True,
    ):
        tab_colors = matplotlib.colors.TABLEAU_COLORS.copy()
        del tab_colors["tab:red"]
        del tab_colors["tab:green"]

        if color_by_particles == False:
            colors = [c for i, c in zip(range(len(prototrack)), cycle(tab_colors))]
        else:
            unique_pids = np.unique(prototrack.particle_id)
            pid_colors = [c for i, c in zip(range(len(unique_pids)), cycle(tab_colors))]
            color_dict = dict(zip(unique_pids, pid_colors))
            colors = [color_dict[pid] for pid in prototrack.particle_id]

        # prototrack
        prototrack["r"] = np.hypot(prototrack.x, prototrack.y)

        pids, counts = np.unique(prototrack.particle_id, return_counts=True)
        maj_pid = pids[np.argmax(counts)]
        pur = max(counts) / len(prototrack)

        try:
            total_meas = self.particles[ self.particles.particle_id == maj_pid ].iloc[0].n_measurements
        except:
            total_meas = "<not found>"

        # make A4 sheet
        fig, ax = self.get_fig_ax(figsize=(8.27, 11.67), ax_config=(2, 1))

        fig.suptitle(
            "Prototrack with ID {} and length {}, "
            "particles: {}, purity: {:.1%}\n"
            "maj pid: {}, total measurements: {}".format(
                str(prototrack.trackId.to_list()[0]), len(prototrack), len(pids), pur, maj_pid, total_meas
            )
        )

        # graph
        thisgraph = self.graph[
            self.graph.edge0.isin(prototrack.measurementId)
            | self.graph.edge1.isin(prototrack.measurementId)
        ]

        for _, row in thisgraph.iterrows():
            try:
                good_edge = (
                    prototrack[prototrack.measurementId == row.edge0].particle_id.iloc[
                        0
                    ]
                    == prototrack[
                        prototrack.measurementId == row.edge1
                    ].particle_id.iloc[0]
                )
                color = "green" if good_edge else "red"
            except:
                print("WARNING: could not verify if edge is good or bad")
                continue
            ax[0].plot(
                [row.z0, row.z1],
                [row.r0, row.r1],
                color=color,
                zorder=-10,
                marker="x",
                alpha=0.7,
            )
            ax[1].plot(
                [row.x0, row.x1],
                [row.y0, row.y1],
                color=color,
                zorder=-10,
                marker="x",
                alpha=0.7,
            )

        # move around a bit so we see overlaps
        f = random_factor

        def randomify(x):
            d = max(x) - min(x)
            r = np.random.uniform(-f * d, f * d, len(x))
            return x + r

        for k in ["x", "y", "z", "r"]:
            prototrack[k] = randomify(prototrack[k])

        ax[0].scatter(prototrack.z, prototrack.r, color=colors, marker="x")
        ax[1].scatter(prototrack.x, prototrack.y, color=colors, marker="x")

        range_vals = {
            "x": (np.inf, 0),
            "y": (np.inf, 0),
            "z": (np.inf, 0),
            "r": (np.inf, 0),
        }

        for c, (_, sp) in zip(colors, prototrack.iterrows()):
            if text:
                ax[0].text(
                    sp.z, sp.r + 5, str(int(sp.measurementId)), c=c, clip_on=True
                )
                ax[1].text(
                    sp.x, sp.y + 5, str(int(sp.measurementId)), c=c, clip_on=True
                )

            for coor in ["x", "y", "z", "r"]:
                range_vals[coor] = min(sp[coor], range_vals[coor][0]), max(
                    sp[coor], range_vals[coor][1]
                )

        def enlarge_range(r):
            d = r[1] - r[0]
            return r[0] - 0.05 * d, r[1] + 0.05 * d

        ax[0].set_xlim(*enlarge_range(range_vals["z"]))
        ax[0].set_ylim(*enlarge_range(range_vals["r"]))
        ax[1].set_xlim(*enlarge_range(range_vals["x"]))
        ax[1].set_ylim(*enlarge_range(range_vals["y"]))

        ax[0].set_xlabel("z")
        ax[0].set_ylabel("r")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")

        fig.tight_layout()
        return fig, ax


# Matching df
match_df = pd.read_csv(snakemake.input[3], dtype={"particle_id": np.uint64})
match_df = match_df[match_df.event == 0].copy()
meff = sum(match_df.matched) / len(match_df.matched)
print(f"matching efficiency: {meff:.2%}")

# Particles
particles = ak.to_dataframe(
    uproot.open(f"{snakemake.input[1]}:particles").arrays(), how="inner"
).reset_index(drop=True)
particles = particles[particles.event_id == 0].copy()
particles = particles[particles.particle_id.isin(match_df.particle_id)].copy()

particles["matched"] = particles.particle_id.map(
    dict(zip(match_df.particle_id, match_df.matched))
)
particles = particles.drop(columns=["vertex_primary", "vertex_secondary", "particle", "generation", "sub_particle"])
assert not any(pd.isna(particles.matched))

# Hits
hits = uproot.open(f"{snakemake.input[2]}:hits").arrays(library="pd")
hits = hits[(hits.event_id == 0) & (hits.tt < 25.0)].copy()
hits["hit_id"] = np.arange(len(hits))

# Digi
simhit_map = pd.read_csv(snakemake.input[4])
measId_to_hitID = dict(zip(simhit_map.measurement_id, simhit_map.hit_id))
hitId_to_particleId = dict(zip(hits.hit_id, hits.particle_id))

# Count overall measurements
measId_particleId_df = pd.DataFrame()
measId_particleId_df["measurement_id"] = simhit_map.measurement_id
measId_particleId_df["particle_id"] = measId_particleId_df["measurement_id"].map({ m: hitId_to_particleId[ measId_to_hitID[m] ] for m in simhit_map.measurement_id })

particle_meas_count_df = measId_particleId_df.groupby("particle_id").count().reset_index()
particles["n_measurements"] = particles.particle_id.map(dict(zip(particle_meas_count_df.particle_id, particle_meas_count_df.measurement_id)))

print(particles.head(5))
print("min measurments: ", min(particles.n_measurements))
print("min pT: ", min(particles.pt))

# Prototracks
prototracks = pd.read_csv(snakemake.input[6])
prototracks["hit_id"] = prototracks["measurementId"].map(measId_to_hitID)
prototracks["tx"] = prototracks.hit_id.map(dict(zip(hits.hit_id, hits.tx)))
prototracks["ty"] = prototracks.hit_id.map(dict(zip(hits.hit_id, hits.ty)))
prototracks["tz"] = prototracks.hit_id.map(dict(zip(hits.hit_id, hits.tz)))
prototracks["geometry_id"] = prototracks.hit_id.map(
    dict(zip(hits.hit_id, hits.geometry_id))
)
prototracks["particle_id"] = prototracks.hit_id.map(hitId_to_particleId)

# Graph
spacepoints = pd.read_csv(snakemake.input[5])
graph = pd.read_csv(snakemake.input[7])

for edge, poscols in [("edge0", ["x0", "y0", "z0"]), ("edge1", ["x1", "y1", "z1"])]:
    for c in poscols:
        graph[c] = graph[edge].map(
            dict(zip(spacepoints.measurement_id, spacepoints[c[:1]]))
        )

graph["r0"] = np.hypot(graph.x0, graph.y0)
graph["r1"] = np.hypot(graph.x1, graph.y1)

# Make interesting collection
particles_not_matched = particles[particles.matched == 0].reset_index()
print("Unmatched particles", len(particles_not_matched))

prototrack_list = [t for _, t in prototracks.groupby("trackId")]

def has_unmatched_particle(t):
    return t.particle_id.isin(particles_not_matched.particle_id).any()

l = len(prototrack_list)
prototrack_list = [t for t in prototrack_list if has_unmatched_particle(t) ]
print(f"Keep only tracks with at least one hit of a unmatched particle: {l} -> {len(prototrack_list)}")

prototrack_list = sorted(prototrack_list, key=lambda t: len(t))
prototrack_list.reverse()

# Plot to pdf
pdf = PdfPages(snakemake.output[0])
plotter = PrototrackPlotter(graph, particles)

plt.rcParams.update({"font.size": 14})

for i in tqdm.tqdm(range(min(50, len(prototrack_list))), desc="make pdf"):
    fig, ax = plotter.plot_prototrack(prototrack_list[i].copy())
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

pdf.close()
