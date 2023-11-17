import numpy as np
import pandas as pd
import re
from pathlib import Path

scores = []
effs = []

for f in snakemake.input:
    f = Path(f)

    match_df = pd.read_csv(f)
    eff = sum(match_df.matched)/len(match_df)

    m = re.match("performance_gnn_plus_ckf_([0-9].[0-9]).csv", f.name)
    assert m
    score = float(m[1])

    print("score:", score, "eff:", eff)
    scores.append(float(m[1]))
    effs.append(eff)

df = pd.DataFrame()

df["score"] = scores
df["eff"] = effs
df["file"] = list(snakemake.input)

df.to_csv(snakemake.output[0])
