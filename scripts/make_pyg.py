import tempfile
import shutil
import yaml
import os
from pathlib import Path
import warnings

import pandas as pd
import awkward as ak
import uproot

from gnn4itk_cf.stages.data_reading.models.acts_reader import ActsReader

class ModifiedActsReader(ActsReader):
    def __init__(self, cfg):
        detector = pd.read_csv(cfg["detector_path"])
        geoid_to_volume = dict(zip(detector.geometry_id, detector.volume_id))

        particles = ak.to_dataframe(uproot.open(f"{cfg['particles_file']}:particles").arrays()).reset_index(drop=True)

        hits = uproot.open(f"{cfg['hit_file']}:hits").arrays(library="pd")
        hits = hits[ hits.tt < 25 ].copy()

        num_events = sum(cfg["data_split"])

        digi_dir = cfg["digi_dir"]
        raw_dir = cfg["input_dir"]

        for i in range(num_events):
            stem = f"event{i:09d}"
            particles[ particles.event_id == i ].to_csv(raw_dir / f"{stem}-particles_initial.csv", index=False)
            hits[ hits.event_id == i ].drop(columns=['volume_id', 'boundary_id', 'layer_id', 'approach_id', 'sensitive_id']).to_csv(raw_dir / f"{stem}-hits.csv", index=False)
            hits = hits.copy()

            measurements = pd.read_csv(digi_dir / f"{stem}-measurements.csv")
            measurements = measurements[ measurements.geometry_id.map(geoid_to_volume).isin([16,17,18]) ].copy()
            measurements.to_csv(raw_dir / f"{stem}-measurements.csv", index=False)

            measurement_simhit_map = pd.read_csv(digi_dir / f"{stem}-measurement-simhit-map.csv")
            measurement_simhit_map = measurement_simhit_map[ measurement_simhit_map.measurement_id.isin(measurements.measurement_id) ]
            measurement_simhit_map.to_csv(raw_dir / f"{stem}-measurement-simhit-map.csv", index=False)

            shutil.copyfile(digi_dir / f"{stem}-cells.csv", raw_dir / f"{stem}-cells.csv")

        super().__init__(cfg)


with tempfile.TemporaryDirectory() as tmp:
    cfg = yaml.load(open(snakemake.input[0], "r"), Loader=yaml.FullLoader)

    cfg["data_split"] = [1,0,0]
    cfg["detector_path"] = snakemake.input[1]
    cfg["particles_file"] = snakemake.input[2]
    cfg["hit_file"] = snakemake.input[3]
    cfg["digi_dir"] = Path(snakemake.input[4]).parent

    raw_dir = Path(tmp) / "raw"
    raw_dir.mkdir()
    cfg["input_dir"] = raw_dir

    stage_dir = Path(tmp) / "stage"
    stage_dir.mkdir()
    cfg["stage_dir"] = stage_dir

    reader = ModifiedActsReader(cfg)
    reader._build_single_csv(reader.raw_events[0], stage_dir)

    csv_events = reader.get_file_names(stage_dir, filename_terms=["particles", "truth"])
    reader._build_single_pyg_event(csv_events[0], stage_dir)

    shutil.copyfile(stage_dir / "event000000000-graph.pyg", snakemake.output[0])
