# Dataset,Quality measure,Beam width,Number of subgroups,Youden
# Facebook,standardQF_1.0,100,3,0.3005369236391844
# brca_n,standardQF_0.5,25,10,1.0
# polluted_synthetic,standardQF_0.5,100,20,0.7579248366013072
# synthetic,standardQF_0.5,250,50,1.0

python3 run.py Facebook standardQF_1.0 100 3 > results/Facebook_time.txt &
python3 run.py brca_n standardQF_0.5 25 10 > results/brca_n_time.txt &
wait
python3 run.py polluted_synthetic standardQF_0.5 100 20 > results/polluted_synthetic_time.txt &
python3 run.py synthetic standardQF_0.5 250 50 > results/synthetic_time.txt

