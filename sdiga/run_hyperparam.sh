# Parameter	Values
# max_generation	250 500 1000
# population_size	10000
# crossover_prob	0.6 0.7 0.9
# mutation_prob	0.01 0.1 
# confidence_weight	0.3 0.4 0.5
# support_weight	0.3 0.4 0.5

# Datasets: Facebook, brca_n, polluted_synthetic, synthetic

# for max_generation in 250 500 1000
# do
    # for crossover_prob in 0.6 0.7 0.9
    # do
    #     for mutation_prob in 0.01 0.1
    #     do
            for confidence_weight in 0.3 0.4 0.5
            do
                for dataset in Facebook brca_n polluted_synthetic synthetic
                do
                    python3 ../main_sdiga.py split_covered_instances SDIGA $dataset --max_generation 500 --population_size 7000 --crossover_prob 0.9 --mutation_prob 0.1 --confidence_weight $confidence_weight --support_weight 0.3 --max_size 10 --num_subgroups 30 > hyperparam_tuning_results/${dataset}_SDIGA_mg${max_generation}_cp${crossover_prob}_mp${mutation_prob}_cw${confidence_weight}_sw0.3.txt &
                    python3 ../main_sdiga.py split_covered_instances SDIGA $dataset --max_generation 500 --population_size 7000 --crossover_prob 0.9 --mutation_prob 0.1 --confidence_weight $confidence_weight --support_weight 0.4 --max_size 10 --num_subgroups 30 > hyperparam_tuning_results/${dataset}_SDIGA_mg${max_generation}_cp${crossover_prob}_mp${mutation_prob}_cw${confidence_weight}_sw0.4.txt &
                    python3 ../main_sdiga.py split_covered_instances SDIGA $dataset --max_generation 500 --population_size 7000 --crossover_prob 0.9 --mutation_prob 0.1 --confidence_weight $confidence_weight --support_weight 0.5 --max_size 10 --num_subgroups 30 > hyperparam_tuning_results/${dataset}_SDIGA_mg${max_generation}_cp${crossover_prob}_mp${mutation_prob}_cw${confidence_weight}_sw0.5.txt &
                    wait
                done
            done
    #     done
    # done
# done
