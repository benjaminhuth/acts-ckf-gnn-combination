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
    parser.add_argument("--modeldir","-m", type=str, default="")
    parser.add_argument("--events", "-n", help="how many events to run", type=int, default=1)
    parser.add_argument("--jobs", "-j", help="parallel jobs", type=int, default=1)
    parser.add_argument("--output", "-o", help="output path", type=str, default="./output")
    parser.add_argument("--output_digi", type=bool, default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--digi", choices=["smear", "truth", "mixed", "mixed-exact"], default="mixed")
    parser.add_argument("--sim", type=str, choices=["fatras", "geant4"], default="geant4")
    parser.add_argument("--finding", type=str, choices=["truth", "gnn"], default="truth")
    parser.add_argument("--input", type=str)
    # fmt: on

    args = vars(parser.parse_args())

    if args["output_digi"]:
        args["outputCsvDigitization"] = os.path.join(args["output"], "digi")
        Path(args["outputCsvDigitization"]).mkdir(exist_ok=True, parents=True)

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

    pipeline.addDefaultCKF()

    if args["finding"] == "truth":
        pipeline.addProofOfConceptTruth()
    else:
        pipeline.addExaTrkX()

    pipeline.addTrackFindingFromPrototracks()

    # as cross check
    pipeline.addTruthTrackingKalman()

    pipeline.run()


if __name__ == "__main__":
    main()
