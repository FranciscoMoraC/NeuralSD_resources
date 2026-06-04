# %%
import sys
sys.path.insert(0, '../')
from neuralsd.iterative_nsd import IterativeNSD
from neuralsd.qc import Qc
from neuralsd.wracc import WRAcc
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import time
import random

# %%
k = 1

# %%
df_train = pd.read_csv("../data/facebook_train.csv")
df_train.head()
df_train = df_train.astype("str")
df_train.columns = [str(i) for i in df_train.columns]
df_test = pd.read_csv("../data/facebook_test.csv")
df_test = df_test.astype("str")
df_test.columns = [str(i) for i in df_test.columns]
# %%
target_tuple = ("target", "government")


qualities = ["WRAcc", "Qc_0.5", "Qc_0.9", "Qc_2.0", "Qc_5.0", "Qc_10.0"]
for q in qualities:
    if q == "WRAcc":
        nsd = IterativeNSD(WRAcc())
        min_quality = 0.01
    else:
        c_value = float(q.split("_")[1])
        nsd = IterativeNSD(Qc(),additional_params_for_quality_measure={"c": c_value})
        min_quality = 10
    # t0 = time.time()
    nsd.fit(df_train, target_tuple, weight_init="best", verbose=False, batch_size=1024, max_iterations=10, min_quality=min_quality)
    # t1 = time.time()
    # print("Time taken:", t1 - t0)

    # %%
    desc = nsd.get_subgroup_descriptions()
    descriptions = [d[0] for d in desc]
    global_entry = pd.Series(False, index=df_test.index)
    for subgroup in descriptions:
        entry = pd.Series(True, index=df_test.index)
        for sel in subgroup:
            attribute = sel[0]
            value = sel[1]
            entry &= (df_test[attribute] == value)
        global_entry |= entry
    target_entry = (df_test[target_tuple[0]] == target_tuple[1])
    positive = sum(global_entry & target_entry)/sum(target_entry)
    negative = sum(global_entry & (~target_entry))/sum(~target_entry)
    youden = positive - negative
    print("Youden", q, youden, flush=True)





