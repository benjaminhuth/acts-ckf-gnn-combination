configfile: "config/config.yaml"


RECO_TYPES = ["standard_ckf", "gnn_plus_ckf", "proof_of_concept", "truth_kalman"]
RECO_TYPES_NO_TT = ["standard_ckf", "gnn_plus_ckf", "proof_of_concept"]
FORMATS = ["csv", "root"]
DIGI_CONFIG_FILE = {
    "125_thickness": "detector/odd-digi-mixed-config-125thickness.json",
    "no_threshold_2": "detector/odd-digi-mixed-config-exact-125thickness.json",
    "high_eff": "detector/odd-digi-mixed-config-125thickness.json",
    "high_eff_no_c": "detector/odd-digi-mixed-config-125thickness.json",
}
CLASSIFIER_CUTS = {
    "125_thickness": [0.5, 0.5],
    "no_threshold_2": [0.5, 0.5],
    "high_eff": [0.05, 0.01, 0.5],
    "high_eff_no_c": [0.05, 0.01, 0.5],
}
CKF_CANDIDATES = {
    "125_thickness": 10,
    "no_threshold_2": 10,
    "high_eff": 10,
    "high_eff_no_c": 1,
}

envvars:
    "CUDA_VISIBLE_DEVICES",


rule simulate_data:
    output:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
    params:
        events=config["n_events"],
        jobs=config["n_sim_jobs"],
    script:
        "scripts/generate_events.py"


rule inference:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "torchscript/{exatrkx_models}/gnn.pt",
    log:
        "tmp/{exatrkx_models}/logs/inference.log",
    output:
        expand(
            "tmp/{{exatrkx_models}}/performance_{type}.{ext}",
            type=RECO_TYPES,
            ext=FORMATS,
        ),
        expand(
            "tmp/{{exatrkx_models}}/seeding_performance_{type}.root",
            type=RECO_TYPES_NO_TT,
        ),
        expand(
            "tmp/{{exatrkx_models}}/tracksummary_{type}.root",
            type=RECO_TYPES_NO_TT,
        ),
        "tmp/{exatrkx_models}/digi/event000000000-measurements.csv",
        "tmp/{exatrkx_models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{exatrkx_models}/digi/event000000000-cells.csv",
        "tmp/{exatrkx_models}/digi/event000000000-spacepoint.csv",
        "tmp/{exatrkx_models}/gnn_plus_ckf/event000000000-exatrkx-graph.csv",
        "tmp/{exatrkx_models}/gnn_plus_ckf/event000000000-prototracks.csv"
    params:
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        digi=lambda wildcards: DIGI_CONFIG_FILE[wildcards.exatrkx_models],
        cuts=lambda wildcards: " ".join(
            [str(c) for c in CLASSIFIER_CUTS[wildcards.exatrkx_models]]
        ),
        ckf_candidates=lambda wildcards: CKF_CANDIDATES[wildcards.exatrkx_models]
    shell:
        "CUDA_VISIBLE_DEVICES={params.cuda_visible_devices} "
        "python3 scripts/gnn_ckf.py -n{config[n_events]} -j{config[n_inference_jobs]} -o tmp/{wildcards.exatrkx_models} -i tmp/simdata "
        "-ckf -km -gnn -poc --digi={params.digi} --modeldir=torchscript/{wildcards.exatrkx_models} "
        "--ckfNCandidates={params.ckf_candidates} "
        "--minEnergyDeposit=3.65e-06 --targetPT=1.0 --cuts {params.cuts} 2>&1 | tee {log}"



rule inference_cpu_for_timing:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "torchscript/{exatrkx_models}/gnn.pt",
    output:
        "tmp/{exatrkx_models}/cpu/timing.tsv",
    params:
        digi=lambda wildcards: DIGI_CONFIG_FILE[wildcards.exatrkx_models],
        cuts=lambda wildcards: " ".join(
            [str(c) for c in CLASSIFIER_CUTS[wildcards.exatrkx_models]]
        ),
    shell:
        "CUDA_VISIBLE_DEVICES='' "
        "python3 scripts/gnn_ckf.py -n3 -j1 -o tmp/{wildcards.exatrkx_models}/cpu -i tmp/simdata "
        "-gnn --digi={params.digi} --modeldir=torchscript/{wildcards.exatrkx_models} "
        "--minEnergyDeposit=3.65e-06 --targetPT=1.0 --cuts {params.cuts}"


rule inference_for_timing:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "torchscript/{exatrkx_models}/gnn.pt",
    log:
        "tmp/{exatrkx_models}/logs/inference_timing.log",
    output:
        "tmp/{exatrkx_models}/timing.tsv",
    params:
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        digi=lambda wildcards: DIGI_CONFIG_FILE[wildcards.exatrkx_models],
        cuts=lambda wildcards: " ".join(
            [str(c) for c in CLASSIFIER_CUTS[wildcards.exatrkx_models]]
        ),
    shell:
        "CUDA_VISIBLE_DEVICES={params.cuda_visible_devices} "
        "python3 scripts/gnn_ckf.py -n3 -j1 -o tmp/{wildcards.exatrkx_models} -i tmp/simdata "
        "-ckf -gnn -poc --digi={params.digi} --modeldir=torchscript/{wildcards.exatrkx_models} "
        "--minEnergyDeposit=3.65e-06 --targetPT=1.0 --cuts {params.cuts} 2>&1 | tee {log}"

rule performance_plots:
    input:
        expand("tmp/{{exatrkx_models}}/performance_{type}.root", type=RECO_TYPES),
    output:
        "plots/{exatrkx_models}/perf_plots.png",
    script:
        "scripts/make_perf_plots.py"

rule performance_plots_ckf_params:
    input:
        "tmp/high_eff/ckf_params/performance_gnn_plus_ckf.root",
        "tmp/high_eff/ckf_params/performance_proof_of_concept.root",
    output:
        "plots/high_eff/ckf_params/perf_plots.png",
    script:
        "scripts/make_perf_plots.py"


rule seeding_plots:
    input:
        expand(
            "tmp/{{exatrkx_models}}/seeding_performance_{type}.root",
            type=RECO_TYPES_NO_TT,
        ),
        expand(
            "tmp/{{exatrkx_models}}/tracksummary_{type}.root",
            type=RECO_TYPES_NO_TT,
        ),
    output:
        "tmp/{exatrkx_models}/seeding_table.csv",
        "plots/{exatrkx_models}/seeding_table.png",
        "plots/{exatrkx_models}/seeding_plot.png",
    script:
        "scripts/make_seeding_plots.py"


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


# Here I use an ugly hack to make cupy find cuda on odslserv01
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
    shell:
        "SNAKEMAKE_INPUT='{input}' "
        "SNAKEMAKE_OUTPUT='{output}' "
        "LD_LIBRARY_PATH='/home/iwsatlas1/bhuth/thirdparty/cuda/lib64' "
        "python3 scripts/make_pyg.py"


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


# Here I use an ugly hack to make cupy find cuda on odslserv01
rule plot_edge_based_metrics:
    input:
        "tmp/{exatrkx_models}/pyg/event000000000-graph.pyg",
        "torchscript/{exatrkx_models}/embedding.pt",
        "torchscript/{exatrkx_models}/filter.pt",
        "torchscript/{exatrkx_models}/gnn.pt",
    output:
        "plots/{exatrkx_models}/filter_gnn_score_hists.png",
        "plots/{exatrkx_models}/edge_metrics_history.png",
        "plots/{exatrkx_models}/scores_pos_neg.png",
        "plots/{exatrkx_models}/edge_eff_eta.png",
    params:
        target_min_hits=3,
        target_min_pt=1.0,
        cuts=lambda wildcards: CLASSIFIER_CUTS[wildcards.exatrkx_models],
    shell:
        "SNAKEMAKE_INPUT='{input}' "
        "SNAKEMAKE_OUTPUT='{output}' "
        "TARGET_MIN_PT='{params.target_min_pt}' "
        "TARGET_MIN_HITS='{params.target_min_hits}' "
        "CUTS='{params.cuts}' "
        "LD_LIBRARY_PATH='/home/iwsatlas1/bhuth/thirdparty/cuda/lib64' "
        "python3 scripts/plot_edge_based_metrics_stages.py"


rule timing_plots:
    input:
        "tmp/{exatrkx_models}/timing.tsv",
        "tmp/{exatrkx_models}/cpu/timing.tsv",
        "tmp/{exatrkx_models}/logs/inference_timing.log",
    output:
        "plots/{exatrkx_models}/timinig_plot.png",
        "plots/{exatrkx_models}/timinig_plot_detail.png",
    script:
        "scripts/plot_timing.py"


#MODELS = ["125_thickness", "no_threshold_2", "high_eff"]
MODELS = ["high_eff"]
MODELSPLUS = ["high_eff", "high_eff_no_c"]


rule cross_perf_plots:
    input:
        expand("tmp/{models}/performance_gnn_plus_ckf.root", models=MODELS),
    output:
        "plots/crosscomp/perf_cross_comparison.png",
    script:
        "scripts/make_perf_cross_comparison.py"


rule cross_perf_plots_high_eff:
    input:
        expand("tmp/{models}/performance_gnn_plus_ckf.root", models=["high_eff", "high_eff_no_c"]),
    output:
        "plots/crosscomp/high_eff_with_without_c_comparison.png",
    script:
        "scripts/make_perf_cross_comparison.py"


rule all:
    default_target: True
    input:
        "plots/crosscomp/perf_cross_comparison.png",
        "plots/crosscomp/high_eff_with_without_c_comparison.png",
        expand("plots/{models}/edge_eff_eta.png", models=MODELS),
        expand("plots/{models}/scores_pos_neg.png", models=MODELS),
        expand("plots/{models}/perf_plots.png", models=MODELS),
        expand("plots/{models}/detailed_matching_hist.png", models=MODELS),
        expand("plots/{models}/detailed_matching_eff.png", models=MODELS),
        expand("plots/{models}/detailed_not_matched_analysis.png", models=MODELS),
        expand("plots/{models}/filter_gnn_score_hists.png", models=MODELS),
        expand("plots/{models}/edge_metrics_history.png", models=MODELS),
        expand("plots/{models}/largest_unmatched_prototracks.pdf", models=MODELS),
        expand("plots/{models}/timinig_plot.png", models=MODELSPLUS),
        expand("plots/{models}/timinig_plot_detail.png", models=MODELSPLUS),
        expand("plots/{models}/seeding_plot.png", models=MODELSPLUS),
