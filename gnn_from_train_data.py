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

from pipeline import Pipeline

u = acts.UnitConstants


def main():
    parser = argparse.ArgumentParser(
        description="Exa.TrkX data generation/reconstruction script"
    )

    # fmt: off
    parser.add_argument("modeldir", type=str)
    parser.add_argument("traindir", type=str)
    parser.add_argument("--events", "-n", help="how many events to run", type=int, default=1)
    parser.add_argument("--jobs", "-j", help="parallel jobs", type=int, default=1)
    parser.add_argument("--output", "-o", help="output path", type=str, default="./output")
    parser.add_argument("--digi", choices=["smear", "geo", "truth", "mixed", "geo-exact"], default="geo-exact")
    # fmt: on

    args = vars(parser.parse_args())
    args["seed"] = 43
    args["input"] = args["traindir"]
    assert Path(args["traindir"]).exists()

    #############################
    # Prepare and run sequencer #
    #############################

    pipeline = Pipeline(args)

    pipeline.readFromFiles()

    # if False:
    #     pipeline.addReader(
    #         acts.examples.CsvMeasurementReader(
    #             level=acts.logging.INFO,
    #             inputDir=args["traindir"],
    #             inputSimHits="simhits",
    #             outputMeasurements="measurements",
    #             outputMeasurementParticlesMap="measurement_particles_map",
    #             outputSourceLinks="sourcelinks",
    #             outputMeasurementSimHitsMap="measurement_simhits_map",
    #             outputClusters="clusters",
    #         )
    #     )
    # else:
    #     pipeline.addDigitization()
    #
    #     pipeline.addAlgorithm(
    #         acts.examples.MakeMeasurementParticlesMap(
    #             level=acts.logging.INFO,
    #             inputSimHits="simhits",
    #             inputMeasurementSimhitMap="measurement_simhits_map",
    #             outputMeasurementParticlesMap="measurement_particles_map",
    #         )
    #     )


    pipeline.addExaTrkX()

    pipeline.addProtoTrackEfficiency()

    pipeline.run()


if __name__ == "__main__":
    main()
