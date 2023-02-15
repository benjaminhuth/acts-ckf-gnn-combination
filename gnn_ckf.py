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
parser.add_argument('--geant', help='Use Geant4 instead of Fatras', action='store_true')
parser.add_argument('--select', help='Select the generated particles', action='store_true')
parser.add_argument('--seed', help='Random seed', type=int, default=42)
parser.add_argument('--generic', help='use generic detector instead of ODD', action='store_true')
parser.add_argument('--digi', help="digitization mode", type=str, choices=['truth', 'geo', 'smear'], default="smear")
args = vars(parser.parse_args())

try:
    assert args['events'] > 0
except:
    parser.print_help()
    exit(1)

outputDir = Path(".")
if not os.path.exists(outputDir):
    outputDir.mkdir()

baseDir = Path(os.path.dirname(__file__))
logger = acts.logging.getLogger("main")


###########################
# Load Open Data Detector #
###########################

acts_root = Path("/home/benjamin/Documents/acts_project/acts")

if args['generic']:
    logger.info("Use generic detector instead of ODD")
    detector, trackingGeometry, decorators = acts.examples.GenericDetector.create()

    if args['digi'] == 'smear':
        digiConfigFile = acts_root / "Examples/Algorithms/Digitization/share/default-smearing-config-generic.json"
    else:
        print("Digitization '{}' mode not supported by generic detector right now".format(args['digi']))
        exit(1)

    geoSelectionSeeding = acts_root / "Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json"
else:
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



#############################
# Prepare and run sequencer #
#############################

field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args['events'],
    numThreads=args['jobs'],
    outputDir=str(outputDir),
)

# Some options
select_particles = False
logger.info("Digitization config file: {}".format(digiConfigFile))
logger.info("Select particles: {}".format(select_particles))


#s = addParticleGun(
    #s,
    #MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, True),
    #EtaConfig(-3.0, 3.0, True),
    #ParticleConfig(1, acts.PdgParticle.eMuon, True),
    #rnd=rnd,
    #multiplicity=50,
#)

s = addPythia8(
    s,
    rnd=rnd,
    outputDirCsv=str(outputDir/"train_all"),
    hardProcess=["HardQCD:all = on"],
    #hardProcess=["Top:qqbar2ttbar=on"],
)

s.addAlgorithm(
    acts.examples.ParticleSelector(
        level=s.config.logLevel,
        inputParticles="particles_input",
        outputParticles="particles_selected",
        removeNeutral=True,
        absEtaMax=3,
        #absEtaMax=1,
        #phiMin=0,
        #phiMax=90 * u.degree,
        rhoMax=2.0 * u.mm,
        ptMin=500 * u.MeV,
        ptMax=20 * u.GeV,
    )
)

s.addAlgorithm(
    acts.examples.FatrasSimulation(
        level=s.config.logLevel,
        inputParticles="particles_selected" if select_particles else "particles_input",
        outputParticlesInitial="particles_initial",
        outputParticlesFinal="particles_final",
        outputSimHits="simhits",
        randomNumbers=rnd,
        trackingGeometry=trackingGeometry,
        magneticField=field,
        generateHitsOnSensitive=True,
    )
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
    selectedParticles="particles_selected" if select_particles else "particles_input",
    outputDirRoot=str(outputDir),
)

################################################
# ExaTrkX (pixels) + KF (pixels) + CKF (other) #
################################################
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

if False:
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
            inputParticles="particles_selected" if select_particles else "particles_input",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputProtoTracks="exatrkx_pixel_prototracks",
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

s.addAlgorithm(
    acts.examples.TrackParamsEstimationAlgorithm(
        level=acts.logging.FATAL,
        inputSpacePoints=["exatrkx_pixel_spacepoints"],
        inputProtoTracks="exatrkx_pixel_prototracks",
        inputSourceLinks="sourcelinks",
        outputProtoTracks="exatrkx_pixel_estimated_prototracks",
        outputTrackParameters="exatrkx_pixel_estimated_parameters",
        trackingGeometry=trackingGeometry,
        magneticField=field,
        #initialVarInflation=[varInflation]*6,
    )
)

s.addAlgorithm(
    acts.examples.TrackFindingFromPrototrackAlgorithm(
        level=acts.logging.INFO,
        inputTracks="exatrkx_pixel_estimated_prototracks",
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputInitialTrackParameters="exatrkx_pixel_estimated_parameters",
        outputTrajectories="final_trajectories",
        outputTrackParameters="final_parameters",
        outputTrackParametersTips="final_tips",
        measurementSelectorCfg=acts.MeasurementSelector.Config(
            [(acts.GeometryIdentifier(), ([], [15.0], [10]))]
        ),
        trackingGeometry=trackingGeometry,
        magneticField=field,
    )
)

s.addWriter(
    acts.examples.CKFPerformanceWriter(
        level=acts.logging.INFO,
        inputParticles="particles_selected" if select_particles else "particles_input",
        inputTrajectories="final_trajectories",
        inputTrackParametersTips="final_tips",
        inputMeasurementParticlesMap="measurement_particles_map",
        # **acts.examples.defaultKWArgs(
        #     # The bottom seed could be the first, second or third hits on the truth track
        #     nMeasurementsMin=CKFPerformanceConfigArg.nMeasurementsMin,
        #     ptMin=CKFPerformanceConfigArg.ptMin,
        #     truthMatchProbMin=CKFPerformanceConfigArg.truthMatchProbMin,
        # ),
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
