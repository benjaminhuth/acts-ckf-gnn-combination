# config: "config/config.yaml"

RECO_TYPES=["standard_ckf", "gnn_plus_ckf", "proof_of_concept", "truth_kalman"]
FORMATS=["csv", "root"]
DIGI={"no_threshold": "mixed-exact"}

envvars:
    "CUDA_VISIBLE_DEVICES"

rule simulate_data:
    output:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
    shell:
        "python3 scripts/generate_events.py -n10 -o tmp/simdata"

rule available_models:
    output:
        "torchscript/no_threshold/gnn.pt"

rule inference:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "torchscript/{exatrkx_models}/gnn.pt",
    output:
        expand("tmp/{{exatrkx_models}}/performance_{type}.{ext}", type=RECO_TYPES, ext=FORMATS),
    params:
        digi=lambda wildcards: DIGI[wildcards.exatrkx_models]
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


rule all:
    default_target: True,
    input:
        "plots/no_threshold/perf_plots.png",
