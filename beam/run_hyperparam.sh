# Datasets: Facebook synthetic polluted_synthetic brca_n
# Measures: chisquaredQF standardQF_0.5 standardQF_1.0
# Beam widths: 10 25 100 250
# Number of subgroups: 3 5 10 20 50
for beam_width in 10 25 100 250; do
    for dataset in brca_n; do
        for measure in chisquaredQF standardQF_0.5 standardQF_1.0; do
            for num_subgroups in 3 5 10 20 50; do
                python3 run_hyperparam.py "$dataset" "$measure" "$beam_width" "$num_subgroups" &
            done
            wait
        done
    done
done
