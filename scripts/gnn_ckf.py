#!/usr/bin/env python3

import sys
import os
import yaml
import pprint
import time
import warnings
import argparse

from typing import Optional, Union
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector
from acts.examples.reconstruction import *
from acts.examples.simulation import *

u = acts.UnitConstants

from pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Exa.TrkX data generation/reconstruction script"
    )

    # fmt: off
    parser.add_argument("--run_proof_of_concept","-poc", action='store_true')
    parser.add_argument("--run_gnn","-gnn", action='store_true')
    parser.add_argument("--run_ckf","-ckf", action='store_true')
    parser.add_argument("--run_truth_kalman","-km", action='store_true')
    parser.add_argument("--modeldir","-m", type=str, default="")
    parser.add_argument("--events", "-n", help="how many events to run", type=int, default=1)
    parser.add_argument("--jobs", "-j", help="parallel jobs", type=int, default=1)
    parser.add_argument("--output", "-o", help="output path", type=str, default="./output")
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--digi", default="mixed")
    parser.add_argument("--sim", type=str, choices=["fatras", "geant4"], default="geant4")
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--minPT", type=float, default=0.5)
    parser.add_argument("--minHits", type=int, default=3)
    parser.add_argument("--minEnergyDeposit", type=float, default=0)
    parser.add_argument("--ensure2EdgesPerVertex", action="store_true")
    parser.add_argument("--useDirectedGraph", action="store_true")
    # fmt: on

    args = vars(parser.parse_args())

    if "input" in args:
        del args["sim"]

    try:
        assert args["events"] > 0
    except:
        parser.print_help()
        exit(1)

    pipeline = Pipeline(args)

    if "input" in args:
        pipeline.readFromFiles()
    else:
        pipeline.addSimulation()

    add_hist_printing = args["events"] == 0

    if args["run_ckf"]:
        pipeline.addDefaultCKF()
    if args["run_proof_of_concept"]:
        pipeline.addProofOfConceptWorkflow()
    if args["run_gnn"]:
        pipeline.addExaTrkXWorkflow(add_eff_printer=True)#add_hist_printing)
    if args["run_truth_kalman"]:
        pipeline.addTruthTrackingKalman()

    pipeline.run()


if __name__ == "__main__":
    main()
