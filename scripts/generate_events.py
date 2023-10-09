from pathlib import Path
import math
import random
import string
import multiprocessing
import subprocess
from functools import partial
import tempfile
import uproot
import os
import numpy as np


def run_geant4_sim(args):
    outputDir, events, skip = args
    outputDir.mkdir(exist_ok=True, parents=True)

    import acts
    import acts.examples

    from acts.examples.odd import getOpenDataDetector
    from acts.examples.simulation import (
        addPythia8,
        addGeant4,
        ParticleSelectorConfig,
    )

    u = acts.UnitConstants

    defaultLogLevel = acts.logging.ERROR

    if not "ODD_DIR" in os.environ:
        for d in [
            Path.home() / "Documents/acts_project/acts",
            Path.home() / "acts",
        ]:
            if d.exists():
                acts_root = d
        oddDir = acts_root / "thirdparty/OpenDataDetector"
    else:
        oddDir = Path(os.environ["ODD_DIR"])
    assert oddDir.exists()

    oddMaterialMap = oddDir / "data/odd-material-maps.root"
    assert oddMaterialMap.exists()

    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    detector, trackingGeometry, decorators = getOpenDataDetector(
        oddDir, mdecorator=oddMaterialDeco, logLevel=defaultLogLevel
    )

    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2 * u.T))
    rnd = acts.examples.RandomNumbers(seed=42)

    outputDir.mkdir(exist_ok=True, parents=True)
    # (outputDir / "csv").mkdir(exist_ok=True, parents=True)

    s = acts.examples.Sequencer(
        events=events,
        skip=skip,
        numThreads=1,
        outputDir=str(outputDir),
        trackFpes=False,
        logLevel=acts.logging.INFO,
    )

    vtxGen = acts.examples.GaussianVertexGenerator(
        stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 0.0 * u.ns),
        mean=acts.Vector4(0, 0, 0, 0),
    )

    addPythia8(
        s,
        vtxGen=vtxGen,
        nhard=1,
        npileup=200,
        rnd=rnd,
        hardProcess=["Top:qqbar2ttbar=on"],
    )

    addGeant4(
        s,
        detector,
        trackingGeometry,
        field,
        preSelectParticles=ParticleSelectorConfig(
            absZ=(0, 1e4),
            rho=(0, 1e3),
            removeNeutral=True,
        ),
        outputDirRoot=outputDir,
        rnd=rnd,
        killVolume=acts.Volume.makeCylinderVolume(r=1.1 * u.m, halfZ=3.0 * u.m),
        killAfterTime=25 * u.ns,
        logLevel=defaultLogLevel,
    )

    s.run()


n_events = snakemake.params["events"]
jobs = min(snakemake.params["jobs"], n_events)
output_dir = Path(snakemake.output[0]).parent
output_dir.mkdir(exist_ok=True, parents=True)
chunks = np.array_split(np.arange(n_events), jobs)

skips = [c[0] for c in chunks]
events = [len(c) for c in chunks]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    outdirs = [tmp / f"subdir{i}" for i in range(jobs)]

    with multiprocessing.Pool(jobs) as p:
        p.map(run_geant4_sim, zip(outdirs, events, skips), 1)

    for filename in [Path(f).name for f in snakemake.output]:
        files = [d / filename for d in outdirs]
        subprocess.run(["hadd", output_dir / filename] + files)

# test consistency
unique_event_ids = np.unique(
    uproot.open(str(output_dir / "hits.root:hits")).arrays(library="pd").event_id
)
assert len(unique_event_ids) == n_events
