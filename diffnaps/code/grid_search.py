import sys
sys.path.append("../")
from utils.experiment_utils import mean_dfs
from utils.utils_base import *
from itertools import *
from utils.utils_base import *
from utils.data_loader import *
from method.diffnaps import *
from utils.data_loader import  perm
from utils.gen_synth_datasets import *
from utils.experiment_utils import *
from utils.measures import *
import time
import argparse

def res_to_csv(method, dataname, res_dict, data,labels,label_dict,translator,verbose=False):
    labels = np.array(labels)
    folder = os.path.join("../results/real_results/",method)
    # print(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    splitted_dataset = split_dataset(data, labels, label_dict,keep=True)
    tsv = "Label \t Pattern \t Pattern_num \t P(class|pattern) \t Supp in class \t Supp in data \n"
    for label_data, data_part in splitted_dataset.items():
        # print("#"*10 + str((label_dict[label_data])) + "#"*10)
        # print("Iter",label_data)
        if not label_data  in res_dict.keys():
            continue
        for pat in res_dict[label_data]:
            supp_1 = np.sum(np.sum(data_part[:,pat],axis=1)==(len(pat)))
            supp_2 = np.sum(np.sum(data[:,pat],axis=1)==(len(pat)))
            rel_supp_1 = supp_1/data_part.shape[0]*100
            rel_supp_2 = supp_2/(data.shape[0])*100
            if rel_supp_2>0.0001:
                odds = rel_supp_1/rel_supp_2
            else:
                odds = 0.0
            if True:
                hit = labels[np.sum(data[:,pat],axis=1)==(len(pat))]==label_data
                if len(hit) > 1 and round(np.sum(hit)/len(hit)*100) > 0:
                    translated = list(map(lambda x: translator[x], pat))
                    if verbose:
                        print(translated)
                        print("Support in class: ", round(rel_supp_1,4),"%")
                        print("Support in data: ",round(rel_supp_2,4),"%")
                        print("Odds: ", round(odds,2))
                        print("P(class=%s|pattern) = "%(label_dict[label_data]), round(np.sum(hit)/len(hit)*100,2), "%")
                        print()
                    strr = label_dict[label_data] + "\t" + str(translated) + "\t" + str(pat)+ "\t" +str(round(np.sum(hit)/len(hit)*100,2))+ "\t" +str(round(rel_supp_1,4)) + "\t" +str(round(rel_supp_2,4))  +"\n" 
                    tsv +=  strr
    # return tsv
    # print(res_dict.keys())
    # with open(os.path.join(folder,dataname+".json"),"w") as f:
    #     json.dump(res_dict, f, indent = 6,cls=NpEncoder)

    with open("results.tsv","w+") as f:
        f.write(tsv)


def get_youden(data):
    results = pd.read_csv("results.tsv", sep="\t")
    results.columns = results.columns.str.strip()

    target_entry = data[target_tuple[0]] == target_tuple[1]
    global_pattern_entry = pd.Series(False, index = data.index)
    for i,r in results.iterrows():
        if r["Label"] != target_tuple[1]:
            continue
        pattern = eval(r["Pattern"])
        pattern_entry = pd.Series(True, index = data.index)
        for p in pattern:
            pattern_entry &= (data[p] == 1)
        global_pattern_entry |= pattern_entry
    positive = (global_pattern_entry & target_entry).sum() / target_entry.sum()
    negative = (global_pattern_entry & ~target_entry).sum() / (~target_entry).sum()
    return positive - negative
    


def get_config(hidden_dim, epochs, weight_decay, lr, batch_size):
    tc = TrainConfig(hidden_dim = hidden_dim, epochs=epochs, weight_decay=weight_decay, elb_k=10, elb_lamb=10, class_elb_k=20, class_elb_lamb=20,
                               lambda_c = 25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=batch_size,
                         log_interval=100, sparse_regu=0, test=False, lr=lr, model=DiffnapsNet,seed=14401360119984179300,init_enc="bimodal")
    tc.t1 = 0.03
    tc.t2 = 0.8
    return tc


grid_params = {
    "hidden_dim": [500, 1000, 2000, 5000, 10000],
    "epochs": [25, 50, 75, 100],
    "weight_decay": [5.0, 8.0, 10.0, 12.0],
    "lr": [0.001, 0.005, 0.01],
    "batch_size": [32, 64, 128]
}

parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset",required=True,type=str)

args = parser.parse_args()
print(args.dataset)

if args.dataset == "facebook":
    data, labels, translator = load_facebook(train = True)
    label_dict = {0:"non-government",1:"government"}
    test_data = pd.read_csv("../../data/facebook_test.csv")
    target_tuple = ("target", "government")
if args.dataset == "synthetic":
    data, labels, translator = load_synthetic(train = True)
    label_dict = {0:"0",1:"1"}
    test_data = pd.read_csv("../../data/synthetic3_test.csv")
    target_tuple = ("Target", 1)
if args.dataset == "polluted_synthetic":
    data, labels, translator = load_synthetic_polluted(train = True)
    label_dict = {0:"0",1:"1"}
    test_data = pd.read_csv("../../data/synthetic3-5_test.csv")
    target_tuple = ("Target", 1)
if args.dataset == "brca_n":
    data, labels, translator = load_brca_bin(train = True)
    label_dict = {0:"0",1:"1"}
    test_data = pd.read_csv("../../data/brca_n_test.csv")
    target_tuple = ("Target", 1)
else:
    raise ValueError("Unknown dataset")

all_results = {}

for param in product(*grid_params.values()):
    conf = get_config(*param)
    model, new_weights, trainDS = learn_diffnaps_net(data, conf, labels=labels, verbose = False)
    enc_w = model.fc0_enc.weight.data.detach().cpu()
    c_w = model.classifier.weight.detach().cpu()
    # extract the differntial patterns, t1 is t_e and t2 is t_c 
    _,_,_,num_pat,res_dict, _ = get_positional_patterns(enc_w,c_w, general=True, t_mean=1.0,  t1=conf.t1,t2=conf.t2)
    res_dict = {int(k):v for k,v in res_dict.items()}
    res_to_csv("diffnaps",args.dataset, res_dict, data, labels, label_dict, translator)
    youden = get_youden(test_data)
    all_results[param] = youden
    print(param, youden, flush=True)
    res = pd.read_csv("results.tsv", sep="\t")
    res.to_csv(os.path.join("../results/real_results/diffnaps/", str(param)+"_"+args.dataset+".tsv"), sep="\t", index=False)
    
print(all_results)
    


