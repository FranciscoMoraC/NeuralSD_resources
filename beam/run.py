# %%
import pysubgroup as ps
import pandas as pd
import sys
sys.path.append("..")
import main
import random
import time

# %%
dataset = sys.argv[1]
measure = sys.argv[2]
beamsize = int(sys.argv[3])
num_subgroups = int(sys.argv[4])

if beamsize < num_subgroups:
    print("Beam size must be greater than or equal to number of subgroups.")
    sys.exit(1)

# %%
if measure.startswith("standardQF"):
    qf_value = float(measure.split("_")[1])
    quality_measure = ps.StandardQF(qf_value)
elif measure == "chisquaredQF":
    quality_measure = ps.ChiSquaredQF()

# %%
df = main.get_data(database=dataset)
target_tuple = main.get_target(database=dataset)

# Mix order of data columns randomly
cols = df.columns.tolist()
random.shuffle(cols)
df = df[cols]


# %%
t0 = time.time()
target = ps.BinaryTarget(target_tuple[0], target_tuple[1])
search_space = ps.create_nominal_selectors(df, ignore=[target_tuple[0]])
task = ps.SubgroupDiscoveryTask(df, target, search_space, result_set_size=num_subgroups, depth=3, qf=quality_measure)
result = ps.BeamSearch(beam_width=beamsize).execute(task)
t1 = time.time()
print(f"Time taken: {t1 - t0}")

# %%
result.to_dataframe().to_csv(f"results/{dataset}.csv", index=False)

