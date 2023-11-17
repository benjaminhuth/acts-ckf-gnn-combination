import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

fig, ax = plt.subplots()

ax.set_title("Track efficiency score sweep")

for f in snakemake.input:
    df = pd.read_csv(f)
    label = str(Path(f).parent).replace("tmp/", "").replace("/score_sweep", "").replace("_", " ")
    ax.plot(df.score, df.eff, "x-", label=label)

ax.set_xlabel("score cut in final GNN")
ax.set_ylabel("tracking efficiency")
ax.set_ylim(0,1)
ax.legend()

fig.tight_layout()
fig.savefig(snakemake.output[0])
