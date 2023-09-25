configfile: "config/config.yaml"


RECO_TYPES = ["standard_ckf", "gnn_plus_ckf", "proof_of_concept", "truth_kalman"]
FORMATS = ["csv", "root"]
DIGI_CONFIG_FILE = {
    "no_threshold": "detector/odd-digi-mixed-config-exact.json",
    "125_thickness": "detector/odd-digi-mixed-config-125thickness.json",
    "no_threshold_2": "detector/odd-digi-mixed-config-exact-125thickness.json",
}
DIGI_MINI_ENERGY_DEPOSIT = {
    "no_threshold": 0.0,
    "125_thickness": 3.65e-06,  # in GeV, 1000 * 3.65 * u.eV
    "no_threshold_2": 3.65e-06,  # in GeV, 1000 * 3.65 * u.eV
}
TARGET_PT = {
    "no_threshold": 0.5,  # GeV
    "125_thickness": 1.0,  # GeV
    "no_threshold_2": 1.0,  # GeV
}


envvars:
    "CUDA_VISIBLE_DEVICES",


rule simulate_data:
    output:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
    params:
        events=20,
        jobs=10,
    shell:
        "scripts/generate_events.py"


rule inference:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "torchscript/{exatrkx_models}/gnn.pt",
    output:
        expand(
            "tmp/{{exatrkx_models}}/performance_{type}.{ext}",
            type=RECO_TYPES,
            ext=FORMATS,
        ),
        "tmp/{exatrkx_models}/digi/event000000000-measurements.csv",
        "tmp/{exatrkx_models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{exatrkx_models}/digi/event000000000-cells.csv",
        "tmp/{exatrkx_models}/digi/event000000000-spacepoint.csv",
        "tmp/{exatrkx_models}/gnn_plus_ckf/event000000000-prototracks.csv",
        "tmp/{exatrkx_models}/timing.tsv",
    params:
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        digi=lambda wildcards: DIGI_CONFIG_FILE[wildcards.exatrkx_models],
        min_energy_deposit=lambda wildcards: DIGI_MINI_ENERGY_DEPOSIT[
            wildcards.exatrkx_models
        ],
        target_pt=lambda wildcards: TARGET_PT[wildcards.exatrkx_models],
    shell:
        "CUDA_VISIBLE_DEVICES={params.cuda_visible_devices} "
        "python3 scripts/gnn_ckf.py -n7 -o tmp/{wildcards.exatrkx_models} -i tmp/simdata "
        "-ckf -km -gnn -poc --digi={params.digi} --modeldir=torchscript/{wildcards.exatrkx_models} "
        "--minEnergyDeposit={params.min_energy_deposit} --minPT={params.target_pt}"


rule performance_plots:
    input:
        expand("tmp/{{exatrkx_models}}/performance_{type}.root", type=RECO_TYPES),
    output:
        "plots/{exatrkx_models}/perf_plots.png",
    script:
        "scripts/perf_plots.py"


rule prototrack_based_plots:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "tmp/{exatrkx_models}/digi/event000000000-measurements.csv",
        "tmp/{exatrkx_models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{exatrkx_models}/gnn_plus_ckf/event000000000-prototracks.csv",
        "tmp/{exatrkx_models}/performance_gnn_plus_ckf.csv",
        "tmp/{exatrkx_models}/performance_proof_of_concept.csv",
    output:
        "plots/{exatrkx_models}/detailed_matching_hist.png",
        "plots/{exatrkx_models}/detailed_matching_eff.png",
        "plots/{exatrkx_models}/detailed_not_matched_analysis.png",
    script:
        "scripts/prototrack_plots.py"


rule make_pyg:
    input:
        "config/data_reading.yaml",
        "config/detectors.csv",
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "tmp/{exatrkx_models}/digi/event000000000-measurements.csv",
        "tmp/{exatrkx_models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{exatrkx_models}/digi/event000000000-cells.csv",
    output:
        "tmp/{exatrkx_models}/pyg/event000000000-graph.pyg",
    script:
        "scripts/make_pyg.py"


rule plot_unmatched_prototracks:
    input:
        "config/detectors.csv",
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "tmp/{exatrkx_models}/performance_gnn_plus_ckf.csv",
        "tmp/{exatrkx_models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{exatrkx_models}/digi/event000000000-spacepoint.csv",
        "tmp/{exatrkx_models}/gnn_plus_ckf/event000000000-prototracks.csv",
        "tmp/{exatrkx_models}/gnn_plus_ckf/event000000000-exatrkx-graph.csv",
    output:
        "plots/{exatrkx_models}/largest_unmatched_prototracks.pdf",
    script:
        "scripts/plot_unmatched_prototracks.py"


rule plot_edge_based_metrics:
    input:
        "tmp/{exatrkx_models}/pyg/event000000000-graph.pyg",
        "torchscript/{exatrkx_models}/embedding.pt",
        "torchscript/{exatrkx_models}/filter.pt",
        "torchscript/{exatrkx_models}/gnn.pt",
    output:
        "plots/{exatrkx_models}/filter_gnn_score_hists.png",
        "plots/{exatrkx_models}/edge_metrics_history.png",
    params:
        target_min_hits=3,
        target_min_pt=lambda wildcards: TARGET_PT[wildcards.exatrkx_models],
    script:
        "scripts/plot_edge_based_metrics_stages.py"


rule timing_plots:
    input:
        "tmp/{exatrkx_models}/timing.tsv",
    output:
        "plots/{exatrkx_models}/timinig_plot.png",
    script:
        "scripts/plot_timing.py"


rule dummy:
    input:
        "plots/no_threshold_2/timinig_plot.png",


# MODELS=["no_threshold", "125_thickness", "no_threshold_2"]
MODELS = [
    "no_threshold_2",
]


rule all:
    default_target: True
    input:
        expand("plots/{models}/perf_plots.png", models=MODELS),
        expand("plots/{models}/detailed_matching_hist.png", models=MODELS),
        expand("plots/{models}/detailed_matching_eff.png", models=MODELS),
        expand("plots/{models}/detailed_not_matched_analysis.png", models=MODELS),
        expand("plots/{models}/filter_gnn_score_hists.png", models=MODELS),
        expand("plots/{models}/edge_metrics_history.png", models=MODELS),
        expand("plots/{models}/largest_unmatched_prototracks.pdf", models=MODELS),
        #expand("plots/{models}/filter_gnn_score_hists.png", models=MODELS),
        #expand("plots/{models}/edge_metrics_history.png", models=MODELS),
