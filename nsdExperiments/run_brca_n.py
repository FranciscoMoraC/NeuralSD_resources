# %%
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import time
import random
import sys
sys.path.insert(0, '../neuralsd')
from iterative_nsd import IterativeNSD

# if not torch.cuda.is_available():
#     raise Exception("GPU not available")

torch.set_default_device("cuda:0")

# %%
k = 1

# %%
df = pd.read_csv("../data/brca_n.csv")
df.head()
df = df.astype("str")
df.columns = [str(i) for i in df.columns]

# %%
target_tuple = ("Target", "1")

# %%
df.shape

# %%
nsd = IterativeNSD(k,"Qc",additional_params_for_quality_measure={"c": 0.5})
t0 = time.time()
nsd.fit(df, target_tuple, weight_init="best", verbose=False, max_iterations=10, min_quality=10)
t1 = time.time()
print("Time taken:", t1 - t0)

# %%


# %%
desc = nsd.get_subgroup_descriptions()
# print("Quality", [quality for _, quality, tp, fp in desc])
# print("tps", [tp for _, quality, tp, fp in desc])
# print("fps", [fp for _, quality, tp, fp in desc])

# %%

# %%
# global_tp = sum([tp for _, _, tp, _ in desc])
# global_fp = sum([fp for _, _, _, fp in desc])
# TP = df[(df[target_tuple[0]] == target_tuple[1])].shape[0]
# FP = df.shape[0] - TP
# youden = (global_tp / TP) - (global_fp / FP)
# print("Youden's J statistic:", youden)

# %%
descriptions = [d[0] for d in desc]
global_entry = pd.Series(False, index=df.index)
num_subgroups = len(descriptions)
sizes = []
prev_entries = []
max_jaccard = 0
for subgroup in descriptions:
    entry = pd.Series(True, index=df.index)
    for sel in subgroup:
        attribute = sel[0]
        value = sel[1]
        entry &= (df[attribute] == value)
    global_entry |= entry
    sizes.append(len(subgroup))
    for prev_entry in prev_entries:
        jaccard = sum(entry & prev_entry)/sum(entry | prev_entry)
        max_jaccard = max(max_jaccard, jaccard)
    prev_entries.append(entry)
target_entry = (df[target_tuple[0]] == target_tuple[1])
positive = sum(global_entry & target_entry)/sum(target_entry)
negative = sum(global_entry & (~target_entry))/sum(~target_entry)
youden = positive - negative
print("Youden's J statistic:", youden)
print("# Subgroups:", num_subgroups)
print("Average subgroup description length:", sum(sizes)/num_subgroups)
print("Max Jaccard similarity between subgroups:", max_jaccard)


