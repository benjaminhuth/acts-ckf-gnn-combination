import numpy as np

configfile: "config/config.yaml"


RECO_TYPES = ["standard_ckf", "gnn_plus_ckf", "proof_of_concept", "truth_kalman"]
RECO_TYPES_NO_TT = ["standard_ckf", "gnn_plus_ckf", "proof_of_concept"]

PIXEL_DIGI="detector/odd-digi-mixed-config-125thickness.json"
PIXEL_SELECTION="detector/odd-geo-selection-pixels.json"

DIGI_CONFIG_FILE = {
    "pixel_only_1GNN": PIXEL_DIGI,
    "pixel_only_2GNN": PIXEL_DIGI,
    "pixel_only_2GNN_neg_weights": PIXEL_DIGI,
    "with_ssbarrel_2GNN": "detector/odd-digi-mixed-config-ssbarrel.json",
}
GEO_SELECTION = {
    "pixel_only_1GNN": PIXEL_SELECTION,
    "pixel_only_2GNN": PIXEL_SELECTION,
    "pixel_only_2GNN_neg_weights": PIXEL_SELECTION,
    "with_ssbarrel_2GNN": "detector/odd-geo-selection-pixels-ssbarrel.json",
}
CLASSIFIER_CUTS = {
    "pixel_only_1GNN": [0.5, 0.5],
    "pixel_only_2GNN": [0.05, 0.01, 0.5],
    "pixel_only_2GNN_neg_weights": [0.05, 0.01, 0.5],
    "with_ssbarrel_2GNN": [0.05, 0.01, 0.5],
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
        lambda wildcards: f"torchscript/{wildcards.models}/gnn.pt",
    log:
        "tmp/{models}/logs/inference.log",
    output:
        expand(
            "tmp/{{models}}/performance_{type}.{ext}",
            type=RECO_TYPES,
            ext=["csv", "root"],
        ),
        expand(
            "tmp/{{models}}/seeding_performance_{type}.root",
            type=RECO_TYPES_NO_TT,
        ),
        expand(
            "tmp/{{models}}/tracksummary_{type}.root",
            type=RECO_TYPES_NO_TT,
        ),
        "tmp/{models}/performance_gnn_plus_ckf_no_c.root",
        "tmp/{models}/digi/event000000000-measurements.csv",
        "tmp/{models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{models}/digi/event000000000-cells.csv",
        "tmp/{models}/digi/event000000000-spacepoint.csv",
        "tmp/{models}/gnn_plus_ckf/event000000000-exatrkx-graph.csv",
        "tmp/{models}/gnn_plus_ckf/event000000000-prototracks.csv"
    params:
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        digi=lambda wildcards: DIGI_CONFIG_FILE[wildcards.models],
        geosel=lambda wildcards: GEO_SELECTION[wildcards.models],
        cuts=lambda wildcards: " ".join(
            [str(c) for c in CLASSIFIER_CUTS[wildcards.models]]
        ),
        ckf_candidates=lambda wildcards: 1 if 'no_c' in wildcards.models else 10,
        models=lambda wildcards: wildcards.models.replace("_no_c", "")
    shell:
        "CUDA_VISIBLE_DEVICES={params.cuda_visible_devices} "
        "python3 scripts/gnn_ckf.py "
        "   -n{config[n_events]} -j{config[n_inference_jobs]} "
        "   -o tmp/{wildcards.models} "
        "   -i tmp/simdata "
        "   -ckf -km -gnn -poc "
        "   --digi={params.digi} "
        "   --gnngeosel={params.geosel} "
        "   --modeldir=torchscript/{params.models} "
        "   --ckfNCandidates={params.ckf_candidates} "
        "   --minEnergyDeposit=3.65e-06 "
        "   --targetPT=1.0 "
        "   --runNoCombinatorics"
        "   --cuts {params.cuts} 2>&1 | tee {log} "

rule inference_score_sweep:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        lambda wildcards: f"torchscript/{wildcards.models}/gnn.pt",
    output:
        "tmp/{models}/score_sweep/performance_gnn_plus_ckf_{score}.csv",
    params:
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        digi=lambda wildcards: DIGI_CONFIG_FILE[wildcards.models],
        geosel=lambda wildcards: GEO_SELECTION[wildcards.models],
        cuts=lambda wildcards: " ".join(
            [ str(c) for c in CLASSIFIER_CUTS[wildcards.models][:-1] ] + [ wildcards.score ]
        ),
        ckf_candidates=lambda wildcards: 1 if 'no_c' in wildcards.models else 10,
        models=lambda wildcards: wildcards.models.replace("_no_c", "")
    shell:
        "CUDA_VISIBLE_DEVICES={params.cuda_visible_devices} "
        "python3 scripts/gnn_ckf.py "
        "   -n1 -j1 "
        "   -o tmp/{wildcards.models}/score_sweep"
        "   -i tmp/simdata "
        "   -gnn "
        "   --digi={params.digi} "
        "   --gnngeosel={params.geosel} "
        "   --modeldir=torchscript/{params.models} "
        "   --ckfNCandidates={params.ckf_candidates} "
        "   --minEnergyDeposit=3.65e-06 "
        "   --targetPT=1.0 "
        "   --cuts {params.cuts} 2>&1 | tee {log}; "
        "mv "
        "    tmp/{wildcards.models}/score_sweep/performance_gnn_plus_ckf.csv "
        "    tmp/{wildcards.models}/score_sweep/performance_gnn_plus_ckf_{wildcards.score}.csv; "
        "(cd tmp/{wildcards.models}/score_sweep && rm -r config.json digi gnn_plus_ckf *.root timing.tsv)"


# rule inference_cpu_for_timing:
#     input:
#         "tmp/simdata/particles_initial.root",
#         "tmp/simdata/hits.root",
#         "torchscript/{models}/gnn.pt",
#     output:
#         "tmp/{models}/cpu/timing.tsv",
#     params:
#         digi=lambda wildcards: DIGI_CONFIG_FILE[wildcards.models],
#         geosel=lambda wildcards: GEO_SELECTION[wildcards.models],
#         cuts=lambda wildcards: " ".join(
#             [str(c) for c in CLASSIFIER_CUTS[wildcards.models]]
#         ),
#     shell:
#         "CUDA_VISIBLE_DEVICES='' "
#         "python3 scripts/gnn_ckf.py -n3 -j1 "
#         "   -o tmp/{wildcards.models}/cpu "
#         "   -i tmp/simdata "
#         "   -gnn "
#         "   --digi={params.digi} "
#         "   --gnngeosel={params.geosel} "
#         "   --modeldir=torchscript/{wildcards.models} "
#         "   --minEnergyDeposit=3.65e-06 "
#         "   --targetPT=1.0 "
#         "   --cuts {params.cuts}"


rule inference_for_timing:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        lambda wildcards: f"torchscript/{wildcards.models}/gnn.pt",
    log:
        "tmp/{models}/logs/inference_timing.log",
    output:
        "tmp/{models}/timing/timing.tsv",
    params:
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        digi=lambda wildcards: DIGI_CONFIG_FILE[wildcards.models],
        geosel=lambda wildcards: GEO_SELECTION[wildcards.models],
        cuts=lambda wildcards: " ".join(
            [str(c) for c in CLASSIFIER_CUTS[wildcards.models]]
        ),
        events=min(config["n_events"], 3),
    shell:
        "CUDA_VISIBLE_DEVICES={params.cuda_visible_devices} "
        "python3 scripts/gnn_ckf.py -n3 -j1 "
        "   -o tmp/{wildcards.models}/timing "
        "   -i tmp/simdata "
        "   -ckf -gnn -poc "
        "   --digi={params.digi} "
        "   --gnngeosel={params.geosel} "
        "   --modeldir=torchscript/{wildcards.models} "
        "   --minEnergyDeposit=3.65e-06 "
        "   --runNoCombinatorics"
        "   --targetPT=1.0 "
        "   --cuts {params.cuts} 2>&1 | tee {log}"



rule performance_plots:
    input:
        expand("tmp/{{models}}/performance_{type}.root", type=RECO_TYPES),
    output:
        "plots/{models}/perf_plots.pdf",
    params:
        with_pt=False,
    script:
        "scripts/make_perf_plots.py"

rule performance_plots_ckf_params:
    input:
        "tmp/high_eff/ckf_params/performance_gnn_plus_ckf.root",
        "tmp/high_eff/ckf_params/performance_proof_of_concept.root",
    output:
        "plots/high_eff/ckf_params/perf_plots.pdf",
    script:
        "scripts/make_perf_plots.py"


rule seeding_plots:
    input:
        expand(
            "tmp/{{models}}/seeding_performance_{type}.root",
            type=RECO_TYPES_NO_TT,
        ),
        expand(
            "tmp/{{models}}/tracksummary_{type}.root",
            type=RECO_TYPES_NO_TT,
        ),
    output:
        "tmp/{models}/seeding_table.csv",
        "plots/{models}/seeding_table.pdf",
        "plots/{models}/seeding_plot.pdf",
    script:
        "scripts/make_seeding_plots.py"


rule prototrack_based_plots:
    input:
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "tmp/{models}/digi/event000000000-measurements.csv",
        "tmp/{models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{models}/gnn_plus_ckf/event000000000-prototracks.csv",
        "tmp/{models}/performance_gnn_plus_ckf.csv",
        "tmp/{models}/performance_proof_of_concept.csv",
    output:
        "plots/{models}/detailed_matching_hist.pdf",
        "plots/{models}/detailed_matching_eff.pdf",
        "plots/{models}/detailed_not_matched_analysis.pdf",
    script:
        "scripts/prototrack_plots.py"


rule particle_type_eff:
    input:
        "tmp/{models}/performance_gnn_plus_ckf.csv",
        "tmp/simdata/particles_initial.root",
    output:
        "latex/{models}_particle_types_eff.tex"
    script:
        "scripts/particle_type.py"


# Here I use an ugly hack to make cupy find cuda on odslserv01
rule make_pyg:
    input:
        "config/data_reading.yaml",
        "config/detectors.csv",
        "tmp/simdata/particles_initial.root",
        "tmp/simdata/hits.root",
        "tmp/{models}/digi/event000000000-measurements.csv",
        "tmp/{models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{models}/digi/event000000000-cells.csv",
    output:
        "tmp/{models}/pyg/event000000000-graph.pyg",
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
        "tmp/{models}/performance_gnn_plus_ckf.csv",
        "tmp/{models}/digi/event000000000-measurement-simhit-map.csv",
        "tmp/{models}/digi/event000000000-spacepoint.csv",
        "tmp/{models}/gnn_plus_ckf/event000000000-prototracks.csv",
        "tmp/{models}/gnn_plus_ckf/event000000000-exatrkx-graph.csv",
    output:
        "plots/{models}/largest_unmatched_prototracks.pdf",
    script:
        "scripts/plot_unmatched_prototracks.py"


# Here I use an ugly hack to make cupy find cuda on odslserv01
rule plot_edge_based_metrics:
    input:
        "tmp/{models}/pyg/event000000000-graph.pyg",
        "torchscript/{models}/embedding.pt",
        "torchscript/{models}/filter.pt",
        "torchscript/{models}/gnn.pt",
    output:
        "plots/{models}/filter_gnn_score_hists.pdf",
        "plots/{models}/edge_metrics_history.pdf",
        "plots/{models}/scores_pos_neg.pdf",
        "plots/{models}/edge_eff_eta.pdf",
    params:
        target_min_hits=3,
        target_min_pt=1.0,
        cuts=lambda wildcards: CLASSIFIER_CUTS[wildcards.models],
    shell:
        "SNAKEMAKE_INPUT='{input}' "
        "SNAKEMAKE_OUTPUT='{output}' "
        "TARGET_MIN_PT='{params.target_min_pt}' "
        "TARGET_MIN_HITS='{params.target_min_hits}' "
        "CUTS='{params.cuts}' "
        "LD_LIBRARY_PATH='/home/iwsatlas1/bhuth/thirdparty/cuda/lib64' "
        "python3 scripts/plot_edge_based_metrics_stages.py"

rule embedding_tsne:
    input:
        "tmp/{models}/pyg/event000000000-graph.pyg",
        "torchscript/{models}/embedding.pt",
    output:
        "plots/{models}/embedding_2D_repr.pdf",
    script:
        "scripts/plot_embedding_TSNE.py"


rule timing_plots_fullchain:
    input:
        "tmp/{models}/timing/timing.tsv",
#         "tmp/{models}/cpu/timing.tsv",
    output:
        "plots/{models}/timinig_plot.pdf",
    script:
        "scripts/plot_timing_full_chain.py"

rule timing_plots_pipeline:
    input:
        "tmp/{models}/timing/timing.tsv",
        "tmp/{models}/logs/inference_timing.log",
    output:
        "plots/{models}/timinig_plot_detail.pdf",
        "plots/{models}/timinig_plot_detail_no_c.pdf",
    script:
        "scripts/plot_timing_pipeline.py"


rule timing_plots_pipeline_crosscomp:
    input:
        "tmp/{modelsA}/timing/timing.tsv",
        "tmp/{modelsA}/logs/inference_timing.log",
        "tmp/{modelsB}/timing/timing.tsv",
        "tmp/{modelsB}/logs/inference_timing.log",
    output:
        "plots/crosscomp/timinig_pipeline_{modelsA}_vs_{modelsB}.pdf",
        "plots/crosscomp/timinig_pipeline_{modelsA}_vs_{modelsB}_no_c.pdf",
    script:
        "scripts/plot_timing_crosscomp_pipelines.py"

rule combine_trackeff_score_sweep:
    input:
        expand("tmp/{{models}}/score_sweep/performance_gnn_plus_ckf_{score}.csv", score=[ str(s)[:3] for s in np.linspace(0,1,11) ]),
    output:
        "tmp/{models}/score_sweep/result.csv",
    script:
        "scripts/combine_trackeff_score_sweep.py"


rule compare_no_combinatorics:
    input:
        "tmp/{models}/performance_gnn_plus_ckf.root",
        "tmp/{models}/performance_gnn_plus_ckf_no_c.root",
    output:
        "plots/{models}/perf_plots_with_without_c.pdf",
    params:
        with_pt=False,
        labels=lambda wildcards: [wildcards.models, f"{wildcards.models} (no combinatorics)"],
        colors=["tab:orange", "tab:purple"],
    script:
        "scripts/make_perf_cross_comparison.py"


MODELS = ["pixel_only_2GNN", "with_ssbarrel_2GNN"]
MODELS_PLUS = ["pixel_only_1GNN", "pixel_only_2GNN", "with_ssbarrel_2GNN"]


rule cross_comparison:
    input:
        expand("tmp/{models}/performance_gnn_plus_ckf.root", models=MODELS_PLUS),
    output:
        "plots/crosscomp/perf_cross_comparison.pdf",
    params:
        with_pt=False,
    script:
        "scripts/make_perf_cross_comparison.py"

rule cross_compare_score_sweep:
    input:
        expand("tmp/{models}/score_sweep/result.csv", models=["pixel_only_2GNN", "pixel_only_1GNN"]),
    output:
        "plots/crosscomp/trackeff_score_sweep.pdf",
    script:
        "scripts/plot_trackeff_score_sweep.py"

rule all:
    default_target: True
    input:
        expand("plots/{models}/edge_eff_eta.pdf", models=MODELS),
#         expand("plots/{models}/scores_pos_neg.pdf", models=MODELS),
        expand("plots/{models}/perf_plots.pdf", models=MODELS_PLUS),
        expand("plots/{models}/detailed_matching_hist.pdf", models=MODELS),
        expand("plots/{models}/detailed_matching_eff.pdf", models=MODELS),
        expand("plots/{models}/detailed_not_matched_analysis.pdf", models=MODELS),
#         expand("plots/{models}/filter_gnn_score_hists.pdf", models=MODELS),
#         expand("plots/{models}/edge_metrics_history.pdf", models=MODELS),
        expand("plots/{models}/largest_unmatched_prototracks.pdf", models=MODELS),
        expand("plots/{models}/timinig_plot.pdf", models=MODELS),
        expand("plots/{models}/timinig_plot_detail.pdf", models=MODELS),
#         expand("plots/{models}/seeding_plot.pdf", models=MODELS),
        expand("plots/{models}/perf_plots_with_without_c.pdf", models=MODELS),
#         expand("latex/{models}_particle_types_eff.tex", models=MODELS),
        "plots/pixel_only_2GNN/embedding_2D_repr.pdf",
        "plots/crosscomp/perf_cross_comparison.pdf",
        f"plots/crosscomp/timinig_pipeline_{MODELS[0]}_vs_{MODELS[1]}.pdf",
#         f"plots/crosscomp/timinig_pipeline_{MODELS[0]}_no_c_vs_{MODELS[1]}_no_c.pdf",
#         "plots/crosscomp/trackeff_score_sweep.pdf",
