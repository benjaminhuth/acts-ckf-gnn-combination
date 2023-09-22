#!/usr/bin/env python3
import sys
import os
import json
import pprint
import time
import warnings
from pathlib import Path

import numpy as np

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector
from acts.examples.reconstruction import *
from acts.examples.simulation import *

u = acts.UnitConstants


class NoneDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        return None


class Pipeline(acts.examples.Sequencer):
    def __init__(self, args):
        args = NoneDict(args.copy())
        self.args = args

        self.outputDir = Path(args["output"])
        self.outputDir.mkdir(parents=True, exist_ok=True)

        if self.args["sim"] is not None and self.args["sim"] != "fatras":
            self.args["jobs"] = 1

        with open(self.outputDir / "config.json", "w") as f:
            json.dump(self.args, f, indent=4)

        pprint.pprint(self.args, indent=4)

        super().__init__(
            events=self.args["events"],
            numThreads=self.args["jobs"],
            outputDir=self.outputDir,
            logLevel=acts.logging.INFO,
            trackFpes=False,
        )

        baseDir = Path(os.path.dirname(__file__)).parent

        ###########################
        # Load Open Data Detector #
        ###########################

        for d in [
            Path("/home/benjamin/Documents/acts_project/acts"),
            Path("/home/iwsatlas1/bhuth/acts"),
        ]:
            if d.exists():
                acts_root = d

        oddDir = acts_root / "thirdparty/OpenDataDetector"
        assert oddDir.exists()

        oddMaterialMap = oddDir / "data/odd-material-maps.root"
        assert oddMaterialMap.exists()

        oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
        self.detector, self.trackingGeometry, decorators = getOpenDataDetector(
            oddDir, mdecorator=oddMaterialDeco
        )

        self.geoSelectionExaTrkX = (
            baseDir / "detector/odd-geo-selection-whole-detector.json"
        )
        assert os.path.exists(self.geoSelectionExaTrkX)

        self.geoSelectionSeeding = oddDir / "config/odd-seeding-config.json"
        assert os.path.exists(self.geoSelectionSeeding)

        self.geoSelectionPixels = baseDir / "detector/odd-geo-selection-pixels.json"
        assert os.path.exists(self.geoSelectionPixels)

        if "digi" in args:
            if Path(args["digi"]).exists():
                self.digiConfigFile = args["digi"]
            elif args["digi"] == "smear":
                self.digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
            elif args["digi"] == "geo":
                self.digiConfigFile = (
                    baseDir / "detector/odd-digi-geometric-config-pixel.json"
                )
            elif args["digi"] == "truth":
                self.digiConfigFile = baseDir / "detector/odd-digi-true-config.json"
            elif args["digi"] == "mixed":
                self.digiConfigFile = baseDir / "detector/odd-digi-mixed-config.json"
            elif args["digi"] == "geo-exact":
                self.digiConfigFile = (
                    baseDir / "detector/odd-digi-geometric-config-pixel-exact.json"
                )
            elif args["digi"] == "mixed-exact":
                self.digiConfigFile = (
                    baseDir / "detector/odd-digi-mixed-config-exact.json"
                )
            else:
                raise RuntimeError(f"unknown digitization type '{args['digi']}'")
            assert os.path.exists(self.digiConfigFile)

        # Target Thresholds
        self.minPT = self.args["minPT"] or 500 * u.MeV
        self.minHits = self.args["minHits"] or 3

        # This selects the tracks we want to look at in the performance plots
        self.targetTrackSelectorConfig = TrackSelectorConfig(
            pt=(500 * u.MeV, None), nMeasurementsMin=3
        )

        # These particles are returned to the chain after the simulation.
        # This means:
        # * When doing truth-track-finding instead of GNN, these are the particles we get seeds from
        # * When doing real inference with GNN, this should only affect the performance writing
        self.targetParticleSelectorConfig = ParticleSelectorConfig(
            removeNeutral=True,
            pt=(self.minPT, None),
            measurements=(self.minHits, None),
            measurementGeometrySelection=acts.examples.readJsonGeometryList(
                str(self.geoSelectionPixels)
            ),
        )

        # This is to avoid problems in simulation
        self.particlePreSelection = ParticleSelectorConfig(
            absZ=(0, 1e4),
            rho=(0, 1e3),
            removeNeutral=True,
        )

        # For both CKFs:
        self.measurementSelectorCfg = acts.MeasurementSelector.Config(
            [(acts.GeometryIdentifier(), ([], [15.0], [10]))]
        )
        # self.measurementSelectorCfg = acts.MeasurementSelector.Config(
        #     [(acts.GeometryIdentifier(), ([], [5.0], [5]))]
        # )

        # Magnetic field
        self.field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

        # Random numbers
        seed = args["seed"] if "seed" in args else 42
        self.rnd = acts.examples.RandomNumbers(seed=seed)

        # Binning cfg for plots
        self.binningCfg = {
            "Pt": acts.examples.Binning("pT [GeV/c]", list(np.logspace(-2, 2, 40))),
            "Eta": acts.examples.Binning("#eta", 40, -4, 4),
            "Phi": acts.examples.Binning("#phi", 100, -3.15, 3.15),
            "Num": acts.examples.Binning("N", 30, -0.5, 29.5),
        }

        # State checkes
        self.hasSimulation = False
        self.hasCKF = False
        self.hasExaTrkxWorkflow = False
        self.hasProofOfConceptWorkflow = False

    def addDigitizationAndParticleSelection(self):
        outputDigi = Path(self.args["output"]) / "digi"
        outputDigi.mkdir(exist_ok=True, parents=True)

        addDigitization(
            self,
            self.trackingGeometry,
            self.field,
            digiConfigFile=self.digiConfigFile,
            outputDirRoot=None,
            outputDirCsv=outputDigi,
            rnd=self.rnd,
            minEnergyDeposit=self.args["minEnergyDeposit"]
        )

        # Make some event data selection for pixels
        self.addAlgorithm(
            acts.examples.SpacePointMaker(
                level=acts.logging.INFO,
                inputSourceLinks="sourcelinks",
                inputMeasurements="measurements",
                outputSpacePoints="pixel_spacepoints",
                trackingGeometry=self.trackingGeometry,
                geometrySelection=acts.examples.readJsonGeometryList(
                    str(self.geoSelectionPixels)
                ),
            )
        )

        self.addWriter(
            acts.examples.CsvSpacepointWriter(
                level=acts.logging.INFO,
                inputSpacepoints="pixel_spacepoints",
                outputDir=outputDigi,
            )
        )

        self.addAlgorithm(
            acts.examples.MeasurementMapSelectorAlgorithm(
                level=acts.logging.INFO,
                inputSourceLinks="sourcelinks",
                inputMeasurementParticleMap="measurement_particles_map",
                outputMeasurementParticleMap="measurement_particles_map_pixels",
                geometrySelection=acts.examples.readJsonGeometryList(
                    str(self.geoSelectionPixels)
                ),
            )
        )

        # Here we use truth seeding as a particle selection
        # Should filter all particles that cannot be seeded properly
        self.addAlgorithm(
            acts.examples.TruthSeedingAlgorithm(
                level=acts.logging.ERROR,
                inputParticles=self.all_particles_key,
                inputMeasurementParticlesMap="measurement_particles_map_pixels",
                inputSpacePoints=["pixel_spacepoints"],
                outputParticles="particle_selection_from_truth_seeding",
                outputProtoTracks="proto_tracks_from_particle_selection",
                outputSeeds="seeds_from_particle_selection",
            )
        )

        self.target_particles_key = "particles_imported_selected"

        # Additionally select particles on momentum, num measurments, ...
        addParticleSelection(
            self,
            self.targetParticleSelectorConfig,
            inputParticles="particle_selection_from_truth_seeding",
            inputMeasurementParticlesMap="measurement_particles_map_pixels",
            inputMeasurements="measurements",
            outputParticles=self.target_particles_key,
            logLevel=acts.logging.DEBUG,
        )

        self.addWhiteboardAlias("particles_selected", self.target_particles_key)

    def readFromFiles(self):
        assert not self.hasSimulation
        self.hasSimulation = True

        inputDir = Path(self.args["input"])

        rootParticlesFile = inputDir / "particles_initial.root"
        rootHitsFile = inputDir / "hits.root"

        self.all_particles_key = "particles_imported"

        if rootParticlesFile.exists() and rootHitsFile.exists():
            self.addReader(
                acts.examples.RootParticleReader(
                    level=acts.logging.DEBUG,
                    particleCollection=self.all_particles_key,
                    filePath=rootParticlesFile,
                )
            )

            self.addReader(
                acts.examples.RootSimHitReader(
                    level=acts.logging.DEBUG,
                    filePath=rootHitsFile,
                    simHitCollection="simhits_imported",
                )
            )
        elif len(list(inputDir.glob("*.csv"))) > 0:
            self.addReader(
                acts.examples.CsvParticleReader(
                    level=acts.logging.DEBUG,
                    inputStem="particles_initial",
                    inputDir=inputDir,
                    outputParticles=self.all_particles_key,
                )
            )

            self.addReader(
                acts.examples.CsvSimHitReader(
                    level=acts.logging.DEBUG,
                    inputStem="hits",
                    inputDir=inputDir,
                    outputSimHits="simhits_imported",
                )
            )
        else:
            raise RuntimeError(
                "found neither root nor csv file in '{}'".format(inputDir)
            )

        self.addAlgorithm(
            acts.examples.HitSelector(
                level=acts.logging.DEBUG,
                inputHits="simhits_imported",
                outputHits="simhits",
                maxTime=25.0 * u.ns,
            )
        )

        if "digi" in self.args:
            self.addDigitizationAndParticleSelection()

    def addSimulation(self):
        assert not self.hasSimulation
        self.hasSimulation = True

        vtxGen = acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 0.0 * u.ns),
            mean=acts.Vector4(0, 0, 0, 0),
        )

        addPythia8(
            self,
            vtxGen=vtxGen,
            rnd=self.rnd,
            outputDirCsv=str(self.outputDir / "train_all"),
            hardProcess=["Top:qqbar2ttbar=on"],
        )

        self.all_particles_key = "all_particles_initial"

        if self.args["sim"] == "fatras":
            addFatras(
                self,
                self.trackingGeometry,
                self.field,
                rnd=self.rnd,
                preSelectParticles=self.particlePreSelection,
                postSelectParticles=None,
                enableInteractions=True,
                outputDirRoot=self.args["outputDirRoot"],
                pMin=1 * u.MeV,
                outputParticlesInitial=self.all_particles_key,
            )
        else:
            addGeant4(
                self,
                self.detector,
                self.trackingGeometry,
                self.field,
                preSelectParticles=self.particlePreSelection,
                postSelectParticles=None,
                outputDirCsv=None,
                outputDirRoot=None,
                rnd=self.rnd,
                killVolume=acts.Volume.makeCylinderVolume(r=1050, halfZ=3000),
                keepParticlesWithoutHits=False,
                outputParticlesInitial=self.all_particles_key,
            )

        self.addAlgorithm(
            acts.examples.HitSelector(
                level=acts.logging.DEBUG,
                inputHits="simhits",
                outputHits="simhits_selected",
                maxTime=25.0 * u.ns,
            )
        )

        self.addWriter(
            acts.examples.RootParticleWriter(
                level=acts.logging.INFO,
                inputParticles=self.all_particles_key,
                filePath=Path(self.args["outputDirRoot"]) / "particles_initial.root",
            )
        )

        self.addWriter(
            acts.examples.RootSimHitWriter(
                level=acts.logging.INFO,
                inputSimHits="simhits_selected",
                filePath=Path(self.args["outputDirRoot"]) / "hits.root",
            )
        )

        if "digi" in self.args:
            self.addDigitizationAndParticleSelection()

    def addDefaultCKF(self):
        assert not self.hasCKF
        self.hasCKF = True

        seedFinderConfig = SeedFinderConfigArg(
            # impactMax=4.426123855748383,
            # deltaR=(13.639924973033985, 50.0854850448914),
            # sigmaScattering=7.3401486140533985,
            # radLengthPerSeed=0.06311548593790932,
            # maxSeedsPerSpM=0,
            cotThetaMax=10.01788, #16.541921673890172,
            # cotThetaMax=27.310 # eta = 4
        )

        addSeeding(
            self,
            self.trackingGeometry,
            self.field,
            truthSeedRanges=None,
            # seedFinderConfigArg=seedFinderConfig,
            geoSelectionConfigFile=self.geoSelectionSeeding,
            outputDirRoot=None,
        )

        # internally converts tracks to trajectories
        addCKFTracks(
            self,
            self.trackingGeometry,
            self.field,
            trackSelectorConfig=self.targetTrackSelectorConfig,
            outputDirRoot=None,
        )

        self.addWriter(
            acts.examples.CKFPerformanceWriter(
                level=acts.logging.ERROR,
                inputParticles=self.target_particles_key,
                inputTrajectories="trajectories",
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=str(self.outputDir / ("performance_standard_ckf.root")),
                effPlotToolConfig=acts.examples.EffPlotToolConfig(self.binningCfg),
                duplicationPlotToolConfig=acts.examples.DuplicationPlotToolConfig(
                    self.binningCfg
                ),
                fakeRatePlotToolConfig=acts.examples.FakeRatePlotToolConfig(
                    self.binningCfg
                ),
            )
        )

    def addProofOfConceptWorkflow(self):
        assert not self.hasProofOfConceptWorkflow
        self.hasProofOfConceptWorkflow = True

        workflow_stem = "proof_of_concept"
        prototrack_key = f"{workflow_stem}_prototracks_before_seeds"
        self.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=acts.logging.INFO,
                inputParticles=self.target_particles_key,
                inputMeasurementParticlesMap="measurement_particles_map_pixels",
                outputProtoTracks=prototrack_key,
            )
        )

        self._addTrackFindingFromPrototracks(prototrack_key, workflow_stem)

    def addExaTrkXWorkflow(self, add_eff_printer=False):
        assert not self.hasExaTrkxWorkflow
        self.hasExaTrkxWorkflow = True

        exatrkxLogLevel = acts.logging.INFO
        modelDir = Path(self.args["modeldir"])

        metricLearningConfig = {
            "level": exatrkxLogLevel,
            "modelPath": modelDir / "embedding.pt",
            "numFeatures": 7,
            "embeddingDim": 8,
            "rVal": 0.2,
            "knnVal": 100,
        }

        filterConfig = {
            "level": exatrkxLogLevel,
            "numFeatures": 3,
            "cut": 0.5,
            "modelPath": modelDir / "filter.pt",
            "nChunks": 0,  # 12,
            "undirected": True,
        }

        gnnConfig = {
            "level": exatrkxLogLevel,
            "numFeatures": 3,
            "cut": 0.5,
            "undirected": True,
            "modelPath": modelDir / "gnn.pt",
        }

        trkConfig = {
          "level": acts.logging.DEBUG,
          "ensure2EdgesPerVertex": self.args["ensure2EdgesPerVertex"] or False,
          "useDirectedGraph": self.args["useDirectedGraph"] or False,
        }

        for cfg in [metricLearningConfig, filterConfig, gnnConfig]:
            assert cfg["modelPath"].exists()

        graphConstructor = acts.examples.TorchMetricLearning(**metricLearningConfig)
        edgeClassifiers = [
            acts.examples.TorchEdgeClassifier(**filterConfig),
            acts.examples.TorchEdgeClassifier(**gnnConfig),
        ]
        trackBuilder = acts.examples.BoostTrackBuilding(**trkConfig)

        workflow_stem = "gnn_plus_ckf"
        prototrack_key = f"{workflow_stem}_prototracks_before_seeds"
        self.addAlgorithm(
            acts.examples.TrackFindingAlgorithmExaTrkX(
                level=exatrkxLogLevel,
                inputSpacePoints="pixel_spacepoints",
                outputProtoTracks=prototrack_key,
                outputGraph="exatrkx_graph",
                inputClusters="clusters",
                inputSimHits="simhits",
                inputParticles=self.all_particles_key,
                inputMeasurementSimhitsMap="measurement_simhits_map",
                graphConstructor=graphConstructor,
                edgeClassifiers=edgeClassifiers,
                trackBuilder=trackBuilder,
                rScale=1000.0,
                phiScale=3.14,
                zScale=3000.0,
                clusterXScale=-1.0,
                clusterYScale=-1.0,
                targetMinPT=self.args["minPT"],
                useGPUsParallel=True,
            )
        )

        csvOutDir = self.outputDir / workflow_stem
        csvOutDir.mkdir(exist_ok=True, parents=True)
        self.addWriter(
            acts.examples.CsvExaTrkXGraphWriter(
                level=acts.logging.INFO,
                inputGraph="exatrkx_graph",
                outputDir=csvOutDir
            )
        )

        self._addTrackFindingFromPrototracks(prototrack_key, workflow_stem)

        if add_eff_printer:
            self._addProtoTrackEfficiency(prototrack_key)

    def _addTrackFindingFromPrototracks(self, prototracks_key, workflow_stem):

        csvOutDir = self.outputDir / workflow_stem
        csvOutDir.mkdir(exist_ok=True, parents=True)

        seed_key = f"{workflow_stem}_seeds_from_prototracks"
        prototrack_after_seed_key = f"{workflow_stem}_exatrkx_prototracks_after_seeds"
        pars_key = f"{workflow_stem}_estimated_parameters"
        self.addAlgorithm(
            acts.examples.PrototracksToParsAndSeeds(
                level=acts.logging.INFO,
                geometry=self.trackingGeometry,
                inputSpacePoints="pixel_spacepoints",
                inputProtoTracks=prototracks_key,
                outputSeeds=seed_key,
                outputProtoTracks=prototrack_after_seed_key,
                outputParameters=pars_key,
            )
        )

        self.addWriter(
            acts.examples.CsvProtoTrackWriter(
                level=acts.logging.INFO,
                inputSpacepoints="pixel_spacepoints",
                inputPrototracks=prototrack_after_seed_key,
                outputDir=csvOutDir,
            )
        )

        self.addWriter(
            acts.examples.CsvTrackParameterWriter(
                level=acts.logging.INFO,
                inputTrackParameters=pars_key,
                outputDir=csvOutDir,
                outputStem="parameters.csv",
            )
        )

        tracks_key = f"{workflow_stem}_final_tracks"
        self.addAlgorithm(
            acts.examples.TrackFindingFromPrototrackAlgorithm(
                level=acts.logging.DEBUG,
                inputProtoTracks=prototrack_after_seed_key,
                inputMeasurements="measurements",
                inputSourceLinks="sourcelinks",
                inputInitialTrackParameters=pars_key,
                outputTracks=tracks_key,
                measurementSelectorCfg=self.measurementSelectorCfg,
                trackingGeometry=self.trackingGeometry,
                magneticField=self.field,
                findTracks=acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction(
                    self.trackingGeometry,
                    self.field,
                    acts.logging.INFO,
                ),
            )
        )

        tracks_selected_key = f"{workflow_stem}_final_tracks_selected"
        addTrackSelection(
            self,
            self.targetTrackSelectorConfig,
            inputTracks=tracks_key,
            outputTracks=tracks_selected_key,
            logLevel=acts.logging.INFO,
        )

        traj_key = f"{workflow_stem}_final_trajectories_selected"
        self.addAlgorithm(
            acts.examples.TracksToTrajectories(
                level=acts.logging.INFO,
                inputTracks=tracks_selected_key,
                outputTrajectories=traj_key,
            )
        )

        self.addWriter(
            acts.examples.CKFPerformanceWriter(
                level=acts.logging.INFO,
                inputParticles=self.target_particles_key,
                inputTrajectories=traj_key,
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=str(
                    self.outputDir / ("performance_" + workflow_stem + ".root")
                ),
                effPlotToolConfig=acts.examples.EffPlotToolConfig(self.binningCfg),
                duplicationPlotToolConfig=acts.examples.DuplicationPlotToolConfig(
                    self.binningCfg
                ),
                fakeRatePlotToolConfig=acts.examples.FakeRatePlotToolConfig(
                    self.binningCfg
                ),
            )
        )

    def _addProtoTrackEfficiency(self, prototracks_key):
        self.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=acts.logging.INFO,
                inputParticles=self.target_particles_key,
                inputMeasurementParticlesMap="measurement_particles_map_pixels",
                outputProtoTracks="truth_prototracks_for_eff",
                minHits=self.minHits,
            )
        )

        self.addAlgorithm(
            acts.examples.ProtoTrackEffPurPrinter(
                level=acts.logging.INFO,
                testProtoTracks=prototracks_key,
                refProtoTracks="truth_prototracks_for_eff",
            )
        )

    def addTruthTrackingKalman(self):
        self.addAlgorithm(
            acts.examples.TruthSeedingAlgorithm(
                level=acts.logging.ERROR,
                inputParticles=self.target_particles_key,
                inputMeasurementParticlesMap="measurement_particles_map",
                inputSpacePoints=["spacepoints"],
                outputParticles="kalman_truth_seeded_particles",
                outputProtoTracks="kalman_truth_seeded_prototracks",
                outputSeeds="kalman_truth_seeds",
            )
        )

        self.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=acts.logging.INFO,
                inputParticles="kalman_truth_seeded_particles",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputProtoTracks="truth_tracking_prototracks_whole_detector",
            )
        )

        self.addAlgorithm(
            acts.examples.TrackParamsEstimationAlgorithm(
                level=acts.logging.ERROR,
                inputSeeds="kalman_truth_seeds",
                outputTrackParameters="kalman_track_parameters",
                trackingGeometry=self.trackingGeometry,
                magneticField=self.field,
                # initialVarInflation=[varInflation]*6,
            )
        )

        kalmanOptions = {
            "multipleScattering": True,
            "energyLoss": True,
            "reverseFilteringMomThreshold": 0.0,
            "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
            "level": acts.logging.WARNING,
        }

        self.addAlgorithm(
            acts.examples.TrackFittingAlgorithm(
                level=acts.logging.WARNING,
                inputMeasurements="measurements",
                inputSourceLinks="sourcelinks",
                inputProtoTracks="truth_tracking_prototracks_whole_detector",
                inputInitialTrackParameters="kalman_track_parameters",
                outputTracks="kalman_truth_tracks",
                pickTrack=-1,
                fit=acts.examples.makeKalmanFitterFunction(
                    self.trackingGeometry, self.field, **kalmanOptions
                ),
                calibrator=acts.examples.makePassThroughCalibrator(),
            )
        )

        addTrackSelection(
            self,
            self.targetTrackSelectorConfig,
            inputTracks="kalman_truth_tracks",
            outputTracks="kalman_truth_tracks_selected",
            logLevel=acts.logging.INFO,
        )

        self.addAlgorithm(
            acts.examples.TracksToTrajectories(
                level=acts.logging.INFO,
                inputTracks="kalman_truth_tracks_selected",
                outputTrajectories="final_trajectories_kalman",
            )
        )

        self.addWriter(
            acts.examples.CKFPerformanceWriter(
                level=acts.logging.WARNING,
                inputParticles=self.target_particles_key,
                inputTrajectories="final_trajectories_kalman",
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=str(self.outputDir / ("performance_truth_kalman.root")),
                effPlotToolConfig=acts.examples.EffPlotToolConfig(self.binningCfg),
                duplicationPlotToolConfig=acts.examples.DuplicationPlotToolConfig(
                    self.binningCfg
                ),
                fakeRatePlotToolConfig=acts.examples.FakeRatePlotToolConfig(
                    self.binningCfg
                ),
            )
        )
