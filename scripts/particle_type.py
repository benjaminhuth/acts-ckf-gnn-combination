import awkward as ak
import numpy as np
import pandas as pd
import uproot

import particle

# Matching df
match_df = pd.read_csv(snakemake.input[0], dtype={"particle_id": np.uint64})

# Particles
particles = ak.to_dataframe(
    uproot.open(f"{snakemake.input[1]}:particles").arrays(), how="inner"
).reset_index(drop=True)


particle_dfs = []

events = np.unique(match_df.event)

for e in events:
    this_match_df = match_df[ match_df.event == e ].copy()
    this_particles = particles[ particles.event_id == e ].copy()

    this_particles = this_particles[ this_particles.particle_id.isin(this_match_df.particle_id) ].copy()

    d = dict(zip(this_match_df.particle_id, this_match_df.matched))
    this_particles["matched"] = this_particles.particle_id.map(d)

    particle_dfs.append(this_particles)

particles = pd.concat(particle_dfs)
print("Electrons")
print(particles[ abs(particles.particle_type) == 11 ])

print("total shape",particles.shape)

particles["abspdg"] = abs(particles.particle_type)
eff = particles.groupby("abspdg").apply(lambda df: pd.DataFrame({"efficiency": [sum(df.matched)/len(df)], "n per event": [len(df)]})).reset_index()
name = eff["abspdg"].map(lambda pdg: particle.Particle.from_pdgid(pdg).pdg_name)
eff.insert(0, "particle", name)
eff = eff.drop(columns=["level_1", "abspdg"])
eff["efficiency"] = eff["efficiency"].map(lambda v: f"{v:.2f}")
eff["n per event"] /= len(events)
eff["n per event"] = eff["n per event"].map(lambda v: round(v))
print(eff)

latex = eff.to_latex(index=False)
with open(snakemake.output[0], 'w') as f:
    f.write(latex)
