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

# %%
k = 1

torch.set_default_device("cuda:0")

# %%
df = pd.read_csv("../data/facebook.csv")
df.head()
df = df.astype("str")
df.columns = [str(i) for i in df.columns]
# Randomly mix the rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
target_tuple = ("target", "government")

# %%
df.shape

# %%
nsd = IterativeNSD(k,"WRAcc")
t0 = time.time()
nsd.fit(df, target_tuple, weight_init="best", verbose=False, batch_size=1024, max_iterations=10, min_quality=0.01)
t1 = time.time()
print("Time taken:", t1 - t0)

# %%
desc = nsd.get_subgroup_descriptions()
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





