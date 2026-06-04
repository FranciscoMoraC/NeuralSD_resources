

python3 ../main_sdiga.py covered_instances SDIGA Facebook --max_generation 500 --population_size 7000 --crossover_prob 0.9 --mutation_prob 0.1 --confidence_weight 0.3 --support_weight 0.5 --max_size 10 --num_subgroups 30 > Facebook_results.txt &
python3 ../main_sdiga.py covered_instances SDIGA synthetic --max_generation 500 --population_size 7000 --crossover_prob 0.9 --mutation_prob 0.1 --confidence_weight 0.5 --support_weight 0.4 --max_size 10 --num_subgroups 30 > synthetic_results.txt &
python3 ../main_sdiga.py covered_instances SDIGA polluted_synthetic --max_generation 500 --population_size 7000 --crossover_prob 0.9 --mutation_prob 0.1 --confidence_weight 0.3 --support_weight 0.4 --max_size 10 --num_subgroups 30 > pollutedSynthetic_results.txt &
wait
python3 ../main_sdiga.py covered_instances SDIGA brca_n --max_generation 500 --population_size 7000 --crossover_prob 0.9 --mutation_prob 0.1 --confidence_weight 0.5 --support_weight 0.4 --max_size 10 --num_subgroups 30 > brca_n_results.txt