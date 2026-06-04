

# grid_parameters = {
#     "redundancyStrategy": ["jaccardSupportDescription", "jaccardSupportDescriptionTarget", "sumJaccard"],
#     "maxRedundancy": [0.4, 0.6, 0.8],
#     "nbIter" : [1000, 10000, 100000],
#     "measure" : ["WRAcc", "WeightedRelativeF1", "RelativeF1", "Acc"]
# }

for redundancyStrategy in "jaccardSupportDescription" "jaccardSupportDescriptionTarget" "sumJaccard"; do
    for maxRedundancy in 0.4 0.6 0.8; do
        for nbIter in 1000 10000 100000; do
            for measure in "WRAcc" "WeightedRelativeF1" "RelativeF1" "Acc"; do
                java -jar MCTS4DM.jar ConfigurationFiles/parameters_${redundancyStrategy}_${maxRedundancy}_${nbIter}_${measure}.conf 
            done
        done
        # wait
    done
done