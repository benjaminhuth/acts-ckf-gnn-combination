#!/usr/bin/env python3

import sys
import os
import yaml
import pprint
import time
import warnings
import argparse

from pathlib import Path

import numpy as np

from pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Exa.TrkX data generation/reconstruction script"
    )

    # fmt: off
    parser.add_argument("--events", "-n", help="how many events to run", type=int, default=1)
    parser.add_argument("--jobs", "-j", help="parallel jobs", type=int, default=1)
    parser.add_argument("--output", "-o", help="output path", type=str, default="./output")
    parser.add_argument("--sim", type=str, choices=["fatras", "geant4"], default="geant4")
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    # fmt: on

    args = vars(parser.parse_args())

    args["outputDirRoot"] = args["output"]
    Path(args["output"]).mkdir(exist_ok=True, parents=True)

    pipeline = Pipeline(args)
    pipeline.addSimulation()
    pipeline.run()


if __name__ == "__main__":
    main()
