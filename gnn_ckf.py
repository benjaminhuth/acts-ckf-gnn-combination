#!/usr/bin/env python3
import sys
import os
import yaml
import pprint
import time
import warnings

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

#########################
# Command line handling #
#########################

import argparse

parser = argparse.ArgumentParser(description='Exa.TrkX data generation/reconstruction script')
parser.add_argument('--events', '-n', help="how many events to run", type=int, default=1)
parser.add_argument('--jobs','-j', help="parallel jobs", type=int, default=1)
parser.add_argument('--output', '-o', help="output path", type=str, default="./output")
parser.add_argument('--seed', help='Random seed', type=int, default=42)
parser.add_argument('--digi', help="digitization mode", type=str, choices=['truth', 'geo', 'smear'], default="smear")
args = vars(parser.parse_args())

try:
    assert args['events'] > 0
except:
    parser.print_help()
    exit(1)

outputDir = Path(args["output"])
outputDir.mkdir(parents=True, exist_ok=True)

baseDir = Path(os.path.dirname(__file__))

###########################
# Load Open Data Detector #
###########################

acts_root = Path("/home/benjamin/Documents/acts_project/acts")

oddDir = acts_root / "thirdparty/OpenDataDetector"

oddMaterialMap = oddDir / "data/odd-material-maps.root"
assert os.path.exists(oddMaterialMap)

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector, trackingGeometry, decorators = getOpenDataDetector(oddDir, mdecorator=oddMaterialDeco)

geoSelectionExaTrkX = baseDir / "detector/odd-geo-selection-whole-detector.json"
geoSelectionSeeding = oddDir / "config/odd-seeding-config.json"
geoSelectionPixels = baseDir / "detector/odd-geo-selection-pixels.json"

if args['digi'] == 'smear':
    digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
elif args['digi'] == 'geo':
    digiConfigFile = baseDir / "detector/odd-digi-geometry-config.json"
elif args['digi'] == 'truth':
    digiConfigFile = baseDir / "detector/odd-digi-true-config.json"

#oddDigiConfigSmear = oddDir / "config/odd-digi-smearing-config.json"
#assert os.path.exists(oddDigiConfigSmear)


assert os.path.exists(digiConfigFile)
# assert os.path.exists(geoSelectionSeeding)
assert os.path.exists(geoSelectionExaTrkX)

# Common CKF Performance config
ckfPerformanceConfig = CKFPerformanceConfig()


#############################
# Prepare and run sequencer #
#############################

logger = acts.logging.getLogger("main")
field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args['events'],
    numThreads=args['jobs'],
    outputDir=str(outputDir),
)

# s = addParticleGun(
#     s,
#     MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, True),
#     EtaConfig(-3.0, 3.0, True),
#     ParticleConfig(4, acts.PdgParticle.eMuon, True),
#     rnd=rnd,
#     multiplicity=200,
# )

s = addPythia8(
    s,
    rnd=rnd,
    outputDirCsv=str(outputDir/"train_all"),
    hardProcess=["HardQCD:all = on"],
    #hardProcess=["Top:qqbar2ttbar=on"],
)

particleSelection = ParticleSelectorConfig(
    rho=(0.0*u.mm, 2.0*u.mm),
    pt=(500*u.MeV, 20*u.GeV),
    absEta=(0, 3)
)

addFatras(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    preSelectParticles=particleSelection,
    outputDirRoot=str(outputDir)
)

s = addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    outputDirRoot=None,
    outputDirCsv=str(outputDir/"train_all"),
    rnd=rnd,
)

######################
# CKF only for check #
######################

seedFinderConfig = SeedFinderConfigArg(
    impactMax = 4.426123855748383,
    deltaR = (13.639924973033985, 50.0854850448914),
    sigmaScattering = 7.3401486140533985,
    radLengthPerSeed = 0.06311548593790932,
    maxSeedsPerSpM = 0,
    cotThetaMax = 16.541921673890172,
    #cotThetaMax=27.310 # eta = 4
)

addSeeding(
    s,
    trackingGeometry,
    field,
    seedFinderConfigArg=seedFinderConfig,
    geoSelectionConfigFile=geoSelectionSeeding,
    outputDirRoot=outputDir,
)

addCKFTracks(
    s,
    trackingGeometry,
    field,
    ckfPerformanceConfig=ckfPerformanceConfig,
    outputDirRoot=str(outputDir),
)

################################################################
# ExaTrkX / TruthTracking (pixels) + KF (pixels) + CKF (other) #
################################################################
if True:
    s.addAlgorithm(
        acts.examples.SpacePointMaker(
            level=acts.logging.INFO,
            inputSourceLinks="sourcelinks",
            inputMeasurements="measurements",
            outputSpacePoints="exatrkx_pixel_spacepoints",
            trackingGeometry=trackingGeometry,
            geometrySelection=acts.examples.readJsonGeometryList(
                str(geoSelectionPixels)
            ),
        )
    )

    exaTrkXConfig = {
        "modelDir": "torchscript",
        "spacepointFeatures": 3,
        "embeddingDim": 8,
        "rVal": 0.2,
        "knnVal": 500,
        "filterCut": 0.01,
        "n_chunks": 5,
        "edgeCut": 0.5,
    }

    logger.info("Exa.TrkX Configuration")
    pprint.pprint(exaTrkXConfig, indent=4)

    s.addAlgorithm(
        acts.examples.TrackFindingAlgorithmExaTrkX(
            level=acts.logging.INFO,
            inputSpacePoints="exatrkx_pixel_spacepoints",
            outputProtoTracks="exatrkx_pixel_prototracks",
            trackFinderML=acts.examples.ExaTrkXTrackFindingTorch(**exaTrkXConfig),
            rScale = 1000.,
            phiScale = np.pi,
            zScale = 1000.,
        )
    )
else:
    s.addAlgorithm(
        acts.examples.TruthTrackFinder(
            level=acts.logging.INFO,
            inputParticles="truth_seeds_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputProtoTracks="exatrkx_pixel_prototracks",
        )
    )

s.addAlgorithm(
    acts.examples.TruthSeedingAlgorithm(
        level=acts.logging.INFO,
        inputSpacePoints=["spacepoints"],
        inputParticles="truth_seeds_selected",
        inputMeasurementParticlesMap="measurement_particles_map",
        outputSeeds="truth-seeds",
        outputParticles="truth_seeds_selected_2",
        outputProtoTracks="seed_proto_track_output",
    )
)

s.addAlgorithm(
    acts.examples.TrackParamsEstimationAlgorithm(
        level=acts.logging.FATAL,
        inputSeeds="truth-seeds",
        outputTrackParameters="exatrkx_pixel_estimated_parameters",
        trackingGeometry=trackingGeometry,
        magneticField=field,
        #initialVarInflation=[varInflation]*6,
    )
)

s.addAlgorithm(
    acts.examples.TrackFindingFromPrototrackAlgorithm(
        level=acts.logging.INFO,
        inputProtoTracks="exatrkx_pixel_prototracks",
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputInitialTrackParameters="exatrkx_pixel_estimated_parameters",
        outputTracks="final_tracks",
        measurementSelectorCfg=acts.MeasurementSelector.Config(
            [(acts.GeometryIdentifier(), ([], [15.0], [10]))]
        ),
        trackingGeometry=trackingGeometry,
        magneticField=field,
        findTracks=acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction(
            trackingGeometry, field
        ),
    )
)

s.addAlgorithm(
    acts.examples.TracksToTrajectories(
        level=acts.logging.INFO,
        inputTracks="final_tracks",
        outputTrajectories="final_trajectories",
    )
)

# This won't work well until we modify also the measurement_particles_map
#s.addWriter(
    #acts.examples.TrackFinderPerformanceWriter(
        #level=acts.logging.INFO,
        #inputProtoTracks="exatrkx_pixel_prototracks",
        #inputParticles="particles_initial",
        #inputMeasurementParticlesMap="measurement_particles_map",
        #filePath=str(outputDir / "track_finding_performance_exatrkx.root"),
    #)
#)

s.addWriter(
    acts.examples.CKFPerformanceWriter(
        level=acts.logging.INFO,
        inputParticles="truth_seeds_selected_2",
        inputTrajectories="final_trajectories",
        inputMeasurementParticlesMap="measurement_particles_map",
        **acts.examples.defaultKWArgs(
            # The bottom seed could be the first, second or third hits on the truth track
            nMeasurementsMin=ckfPerformanceConfig.nMeasurementsMin,
            ptMin=ckfPerformanceConfig.ptMin,
            truthMatchProbMin=ckfPerformanceConfig.truthMatchProbMin,
        ),
        filePath=str(outputDir / "performance_kf_plus_ckf.root"),
    )
)

#s.addAlgorithm(
    #acts.examples.TrajectoriesToPrototracks(
        #level=acts.logging.INFO,
        #inputTrajectories="final_trajectories",
        #outputPrototracks="final_prototracks",
    #)
#)

#s.addWriter(
    #acts.examples.TrackFinderPerformanceWriter(
        #level=acts.logging.INFO,
        #inputProtoTracks="final_prototracks",
        #inputParticles="particles_initial",
        #inputMeasurementParticlesMap="measurement_particles_map",
        #filePath=str(outputDir / "track_finding_performance_kf_plus_ckf.root"),
    #)
#)
        
s.run()
