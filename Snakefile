# config: "config/config.yaml"

RECO_TYPES = ["standard_ckf", "gnn_plus_ckf", "proof_of_concept", "truth_kalman"]
FORMATS = ["csv", "root"]
DIGI = {"no_threshold": "mixed-exact"}


envvars:
    "CUDA_VISIBLE_DEVICES",


rule simulate_data:
    output:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
    shell:
        "python3 scripts/generate_events.py -n10 -o tmp/simdata"


rule available_models:
    output:
        "torchscript/no_threshold/gnn.pt",


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
    params:
        digi=lambda wildcards: DIGI[wildcards.exatrkx_models],
    shell:
        "python3 scripts/gnn_ckf.py -n1 -o tmp/{wildcards.exatrkx_models} -i tmp/simdata "
        "-ckf -km -gnn -poc --digi={params.digi} --modeldir=torchscript/{wildcards.exatrkx_models}"


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


rule all:
    default_target: True
    input:
        "plots/no_threshold/perf_plots.png",
        "plots/no_threshold/detailed_matching_hist.png",
        "plots/no_threshold/detailed_matching_eff.png",
        "plots/no_threshold/detailed_not_matched_analysis.png",
