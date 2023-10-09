#!/usr/bin/env python3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = snakemake.params.cuda_visible_devices

import acts

u = acts.UnitConstants

from pipeline import Pipeline

args = {
    "events": snakemake.params.events,
    "output": snakemake.params.outdir,
    "input": snakemake.params.indir,
    "modeldir": snakemake.params.modeldir,
    "minPT": 1.0 * u.GeV,
    "minHits": 3,
    "minEnergyDeposit": 3.65e-06,
    "digi": snakemake.params.digi,
    "seed": 42,
    "jobs": 1,
}

pipeline = Pipeline(args)
pipeline.readFromFilesAndDigitize()

pipeline.addDefaultCKF()
pipeline.addProofOfConceptWorkflow()
pipeline.addExaTrkXWorkflow()
pipeline.addTruthTrackingKalman()

pipeline.run()
