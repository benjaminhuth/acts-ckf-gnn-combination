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


def main():
    parser = argparse.ArgumentParser(
        description="Exa.TrkX data generation/reconstruction script"
    )

    # fmt: off
    parser.add_argument("--events", "-n", help="how many events to run", type=int, default=1)
    parser.add_argument("--jobs", "-j", help="parallel jobs", type=int, default=1)
    parser.add_argument("--output", "-o", help="output path", type=str, default="./output")
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--digi", choices=["smear", "geo", "truth"], default="smear")
    parser.add_argument("--sim", type=str, choices=["fatras", "geant4"], default="geant4")
    parser.add_argument("--finding", type=str, choices=["truth", "gnn"], default="truth")
    # fmt: on

    args = vars(parser.parse_args())

    try:
        assert args["events"] > 0
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
    detector, trackingGeometry, decorators = getOpenDataDetector(
        oddDir, mdecorator=oddMaterialDeco
    )

    geoSelectionExaTrkX = baseDir / "detector/odd-geo-selection-whole-detector.json"
    geoSelectionSeeding = oddDir / "config/odd-seeding-config.json"
    geoSelectionPixels = baseDir / "detector/odd-geo-selection-pixels.json"

    if args["digi"] == "smear":
        digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
    elif args["digi"] == "geo":
        digiConfigFile = baseDir / "detector/odd-digi-geometry-config.json"
    elif args["digi"] == "truth":
        digiConfigFile = baseDir / "detector/odd-digi-true-config.json"

    # oddDigiConfigSmear = oddDir / "config/odd-digi-smearing-config.json"
    # assert os.path.exists(oddDigiConfigSmear)

    assert os.path.exists(digiConfigFile)
    # assert os.path.exists(geoSelectionSeeding)
    assert os.path.exists(geoSelectionExaTrkX)

    # This selects the tracks we want to look at in the performance plots
    targetTrackSelectorConfig = TrackSelectorConfig(
        pt=(500 * u.MeV, None), nMeasurementsMin=3
    )

    # These particles are returned to the chain after the simulation.
    # This means:
    # * When doing truth-track-finding instead of GNN, these are the particles we get seeds from
    # * When doing real inference with GNN, this should only affect the performance writing
    targetParticleSelectorConfig = ParticleSelectorConfig(
        pt=(500 * u.MeV, None),
        removeNeutral=True,
    )

    # This is to avoid problems in simulation
    particlePreSelection = ParticleSelectorConfig(
        absZ=(0, 1e4),
        rho=(0, 1e3),
        removeNeutral=True,
    )

    #############################
    # Prepare and run sequencer #
    #############################

    logger = acts.logging.getLogger("main")
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
    rnd = acts.examples.RandomNumbers(seed=42)

    s = acts.examples.Sequencer(
        events=args["events"],
        numThreads=args["jobs"] if args["sim"] == "fatras" else 1,
        outputDir=str(outputDir),
        logLevel=acts.logging.INFO,
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
        outputDirCsv=str(outputDir / "train_all"),
        hardProcess=["HardQCD:all = on"],
        # hardProcess=["Top:qqbar2ttbar=on"],
    )

    if args["sim"] == "fatras":
        addFatras(
            s,
            trackingGeometry,
            field,
            rnd=rnd,
            preSelectParticles=particlePreSelection,
            postSelectParticles=targetParticleSelectorConfig,
            outputDirRoot=str(outputDir),
        )
    else:
        addGeant4(
            s,
            detector,
            trackingGeometry,
            field,
            preSelectParticles=particlePreSelection,
            postSelectParticles=targetParticleSelectorConfig,
            outputDirCsv=str(outputDir / "train_all"),
            outputDirRoot=None,
            rnd=rnd,
            killVolume=acts.Volume.makeCylinderVolume(r=1050, halfZ=3000),
            keepParticlesWithoutHits=False,
        )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        outputDirRoot=None,
        outputDirCsv=str(outputDir / "train_all"),
        rnd=rnd,
    )
    
#     addAmbiguityResolution(
#         
#     )

    ######################
    # CKF for comparison #
    ######################

    seedFinderConfig = SeedFinderConfigArg(
        impactMax=4.426123855748383,
        deltaR=(13.639924973033985, 50.0854850448914),
        sigmaScattering=7.3401486140533985,
        radLengthPerSeed=0.06311548593790932,
        maxSeedsPerSpM=0,
        cotThetaMax=16.541921673890172,
        # cotThetaMax=27.310 # eta = 4
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
        outputDirRoot=str(outputDir),
        trackSelectorConfig=targetTrackSelectorConfig,
    )

    ################################################################
    # ExaTrkX / TruthTracking (pixels) + KF (pixels) + CKF (other) #
    ################################################################

    if args["finding"] == "gnn":
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
                rScale=1000.0,
                phiScale=np.pi,
                zScale=1000.0,
            )
        )
            
        s.addAlgorithm(
            acts.examples.PrototracksToSeeds(
                level=acts.logging.INFO,
                inputSpacePoints="exatrkx_pixel_spacepoints"
                if args["finding"] == "gnn"
                else "spacepoints",
                inputProtoTracks="exatrkx_pixel_prototracks",
                outputSeeds="exatrkx_seeds",
                outputProtoTracks="exatrkx_pixel_prototracks_after_seeds",
            )
        )
    else:
        print("WARNING: Use truth tracking for Pixels!")
        s.addAlgorithm(
            acts.examples.MeasurementMapSelectorAlgorithm(
                level=acts.logging.INFO,
                inputSourceLinks="sourcelinks",
                inputMeasurementParticleMap="measurement_particles_map",
                outputMeasurementParticleMap="measurement_particles_map_pixels",
                geometrySelection=acts.examples.readJsonGeometryList(
                    str(geoSelectionPixels)
                ),
            )
        )

        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=acts.logging.INFO,
                inputParticles="particles_selected",
                inputMeasurementParticlesMap="measurement_particles_map_pixels",
                outputProtoTracks="exatrkx_pixel_prototracks_after_seeds",
            )
        )
            
        s.addAlgorithm(
            acts.examples.TruthSeedingAlgorithm(
                level=acts.logging.INFO,
                inputParticles="particles_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                inputSpacePoints=["spacepoints"],
                outputParticles="truth_seeded_particles",
                outputProtoTracks="truth_particle_tracks",
                outputSeeds="exatrkx_seeds",
            )
        )

    s.addAlgorithm(
        acts.examples.TrackParamsEstimationAlgorithm(
            level=acts.logging.FATAL,
            inputSeeds="exatrkx_seeds",
            outputTrackParameters="exatrkx_pixel_estimated_parameters",
            trackingGeometry=trackingGeometry,
            magneticField=field,
            # initialVarInflation=[varInflation]*6,
        )
    )

    s.addAlgorithm(
        acts.examples.TrackFindingFromPrototrackAlgorithm(
            level=acts.logging.INFO,
            inputProtoTracks="exatrkx_pixel_prototracks_after_seeds",
            inputMeasurements="measurements",
            inputSourceLinks="sourcelinks",
            inputInitialTrackParameters="exatrkx_pixel_estimated_parameters",
            outputTracks="final_tracks",
            measurementSelectorCfg=acts.MeasurementSelector.Config(
                [(acts.GeometryIdentifier(), ([], [15.0], [10]))]
            ),  # should be the same as in addCKFTracks
            trackingGeometry=trackingGeometry,
            magneticField=field,
            findTracks=acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction(
                trackingGeometry,
                field,
                acts.logging.INFO,
            ),
        )
    )

    addTrackSelection(
        s,
        targetTrackSelectorConfig,
        inputTracks="final_tracks",
        outputTracks="final_tracks_selected",
        logLevel=acts.logging.INFO,
    )

    s.addAlgorithm(
        acts.examples.TracksToTrajectories(
            level=acts.logging.INFO,
            inputTracks="final_tracks_selected",
            outputTrajectories="final_trajectories_selected",
        )
    )

    s.addWriter(
        acts.examples.CKFPerformanceWriter(
            level=acts.logging.INFO,
            inputParticles="particles_initial_selected",
            inputTrajectories="final_trajectories_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "performance_kf_plus_ckf.root"),
        )
    )

    # This won't work well until we modify also the measurement_particles_map
    # s.addWriter(
    # acts.examples.TrackFinderPerformanceWriter(
    # level=acts.logging.INFO,
    # inputProtoTracks="exatrkx_pixel_prototracks",
    # inputParticles="particles_initial",
    # inputMeasurementParticlesMap="measurement_particles_map",
    # filePath=str(outputDir / "track_finding_performance_exatrkx.root"),
    # )
    # )

    # s.addAlgorithm(
    # acts.examples.TrajectoriesToPrototracks(
    # level=acts.logging.INFO,
    # inputTrajectories="final_trajectories",
    # outputPrototracks="final_prototracks",
    # )
    # )

    # s.addWriter(
    # acts.examples.TrackFinderPerformanceWriter(
    # level=acts.logging.INFO,
    # inputProtoTracks="final_prototracks",
    # inputParticles="particles_initial",
    # inputMeasurementParticlesMap="measurement_particles_map",
    # filePath=str(outputDir / "track_finding_performance_kf_plus_ckf.root"),
    # )
    # )

    s.run()


if __name__ == "__main__":
    main()
