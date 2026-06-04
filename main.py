from subgroups.algorithms.subgroup_sets.idsd import IDSD
from subgroups.algorithms.subgroup_sets.bsd import BSD
from subgroups.algorithms.subgroup_sets.qfinder import QFinder
from subgroups.algorithms.subgroup_sets.sdmapstar import SDMapStar
from subgroups.algorithms.subgroup_sets.vlsd import VLSD
from subgroups.quality_measures.wracc import WRAcc
from subgroups.quality_measures.wracc_optimistic_estimate_1 import WRAccOptimisticEstimate1
from subgroups.quality_measures.binomial_test import BinomialTest
from subgroups.quality_measures.binomial_test_optimistic_estimate_1 import BinomialTestOptimisticEstimate1
from subgroups.quality_measures.piatetsky_shapiro import PiatetskyShapiro
from subgroups.quality_measures.piatetsky_shapiro_optimistic_estimate_1 import PiatetskyShapiroOptimisticEstimate1
from subgroups import datasets


from pandas import DataFrame,read_csv, Series, concat
import os
import time
import random 
import psutil
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse

quality_models = ["BSD","SDMapStar","VLSD"]

## ALL TUNING PARAMETERS

params = {
    "num_subgroups" : 10,
    "max_size" : 5,
    "coverage_thld" : 0.1,
    "ppv_thld" : 0.5,
    "or_thld" : 1.2,
    "p_val_thld" : 0.05,
    "abs_contribution_thld" : 0.2,
    "contribution_thld" : 5,
    "min_rank" : 3,
    "folds" : 10,
    "instances" : None,
    "columns" : None,
    "split" : False,
    "validation" : False,
    "quality_measure" : WRAcc(),
    "optimistic_estimate" : WRAccOptimisticEstimate1()
}

def get_model(model, write,database = None):
    id = random.randint(0, 100000)
    if database is not None and write:
        if model == "SDMapStar":
            if database == "MIMIC":
                thresh = .01
                min_n = 100
            elif database == "Mushrooms":
                thresh = .1
                min_n = 100
            elif database == "AB":
                thresh = -1
                min_n = 0
            elif database == "HealthyAging":
                thresh = 0.01
                min_n = 50
            elif database == "sick":
                thresh = 0.05
                min_n = 50
            elif database == "Soybean":
                if params["quality_measure"].get_name() == "WRAcc":
                    thresh = 0.1
                    if params["split"]:
                        min_n = 57
                    elif params["validation"]:
                        min_n = 49
                        thresh = 0.09
                    else:
                        min_n = 70
                elif params["quality_measure"].get_name() == "BinomialTest":
                    if params["split"] or params["validation"]:
                        thresh = -1
                        min_n = 0
                    else:
                        thresh = 5
                        min_n = 30
                elif params["quality_measure"].get_name() == "PiatetskyShapiro":
                    thresh = 0.1
                    if params["split"] or params["validation"]:
                        min_n = 57
                    else:
                        min_n = 70
            elif database == "StudentSuccess":
                if params["quality_measure"].get_name() == "PiatetskyShapiro":
                    thresh = 0.05
                    min_n = 50
                elif params["quality_measure"].get_name() == "WRAcc":
                    thresh = 0.06
                    min_n = 100
                elif params["quality_measure"].get_name() == "BinomialTest":
                    thresh = 0.05
                    min_n = 50
            else:
                print("WARNING: LIMIT NOT TESTED")
                thresh = 0
                min_n = 0
        elif model == "VLSD":
            thresh = 0.05
    
    if database is not None:
        output_file = f"results_{model}_{id}_{database}.txt"
    else:
        output_file = f"results_{model}_{id}.txt"

    if model == "IDSD":
        # return IDSD(num_subgroups=10, max_complexity=5, write_results_in_file=write, file_path=output_file)
        return IDSD(num_subgroups=params["num_subgroups"], max_complexity=params["max_size"], write_results_in_file=write, file_path=output_file,
                    coverage_thld = params["coverage_thld"], or_thld = params["or_thld"], p_val_thld = params["p_val_thld"], abs_contribution_thld = params["abs_contribution_thld"], contribution_thld = params["contribution_thld"]), id
    elif model == "BSD":
        return BSD(quality_measure=params["quality_measure"], optimistic_estimate=params["optimistic_estimate"], min_support=0, num_subgroups=params["num_subgroups"], max_depth=params["max_size"]-1, write_results_in_file=write, file_path=output_file), id
    elif model == "QFinder":
        return QFinder(num_subgroups=params["num_subgroups"], max_complexity=params["max_size"], write_results_in_file=write, file_path=output_file,
                       coverage_thld = params["coverage_thld"], or_thld = params["or_thld"], p_val_thld = params["p_val_thld"], abs_contribution_thld = params["abs_contribution_thld"], contribution_thld = params["contribution_thld"]), id
    elif model == "SDMapStar":
        if write:
            return SDMapStar(quality_measure=params["quality_measure"], optimistic_estimate=params["optimistic_estimate"],minimum_quality_measure_value = thresh, minimum_n=min_n, num_subgroups=params["num_subgroups"] +2, write_results_in_file=True, file_path=output_file), id
        else:
            return SDMapStar(quality_measure=params["quality_measure"], optimistic_estimate=params["optimistic_estimate"],minimum_quality_measure_value = -2, minimum_n=0, num_subgroups=params["num_subgroups"], write_results_in_file=False), id
    elif model == "VLSD":
        if write:
            return VLSD(quality_measure=params["quality_measure"], optimistic_estimate=params["optimistic_estimate"], q_minimum_threshold=thresh, oe_minimum_threshold=thresh, write_results_in_file=True, file_path=output_file), id
        else:
            return VLSD(quality_measure=params["quality_measure"], optimistic_estimate=params["optimistic_estimate"], q_minimum_threshold=-2, oe_minimum_threshold=-2, write_results_in_file=False), id
    elif model == "BerryFinder":
        return BerryFinder(max_complexity=params["max_size"], write_results_in_file=write, file_path=output_file, min_rank = params["min_rank"], 
                   coverage_thld = params["coverage_thld"], ppv_thld = params["ppv_thld"], or_thld = params["or_thld"], p_val_thld = params["p_val_thld"], abs_contribution_thld = params["abs_contribution_thld"], contribution_thld = params["contribution_thld"]
                   ), id
    else:
        raise ValueError("Model not recognized")

def get_data(database:str):
    if database == "MIMIC":
        return get_mimic_data()
    elif database == "Mushrooms":
        return get_mushrooms_data()
    elif database == "Vancomycin":
        return get_vancomycin_data()
    elif database == "Chess":
        return get_chess_data()
    elif database == "Soybean":
        return get_soybean_data()
    elif database == "HealthyAging":
        return get_healthy_aging_data()
    elif database == "sick":
        return get_sick_data()
    elif database == "credit":
        return get_credit_data()
    elif database == "connect4":
        return get_connect4_data()
    elif database == "StudentSuccess":
        return get_student_success_data()
    elif database == "Car":
        return get_car_data()
    elif database == "Dermatology":
        return get_dermatology_data()
    elif database == "Depression":
        return get_depression_data()
    elif database == "MIMIC-IV":
        return get_mimic_iv_data()
    elif database == "Cardio":
        return get_cardio_data()
    elif database == "Facebook":
        return get_facebook_data()
    elif database == "synthetic":
        return get_synthetic_data()
    elif database == "polluted_synthetic":
        return get_polluted_synthetic_data()
    elif database == "brca_n":
        return get_brca_n_data()
    else:
        raise ValueError("Database not recognized")
    
def get_folds(dataframe: DataFrame, target:tuple):
    skf = StratifiedKFold(n_splits=params["folds"], random_state=42, shuffle=True)
    folds = []
    for train_index, test_index in skf.split(dataframe, dataframe[target[0]]):
        train = dataframe.iloc[train_index]
        test = dataframe.iloc[test_index]
        folds.append((train, test))
    return folds

def get_mimic_data():
    if params["split"]:
        df_train = read_csv('../data/mimic-iii-train.csv')
        df_test = read_csv('../data/mimic-iii-test.csv')
        df_train = df_train.astype("str")
        df_train.columns = df_train.columns.astype(str)
        df_test = df_test.astype("str")
        df_test.columns = df_test.columns.astype(str)
        return df_train, df_test
    df = read_csv('../data/mimic-iii.csv')
    df = df.astype("str")
    df.columns = df.columns.astype(str)
    return df

def get_mushrooms_data():
    if params["split"]:
        df_train = read_csv('../data/agaricus-lepiota-train.csv')
        df_test = read_csv('../data/agaricus-lepiota-test.csv')
        df_train = df_train.astype("str")
        df_train.columns = [ "poisonous",
        "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color",
        "population", "habitat"]
        df_test = df_test.astype("str")
        df_test.columns = df_train.columns
        # Remove veil-type column (only one value)
        df_train.drop(columns=["veil-type"], inplace=True)
        df_test.drop(columns=["veil-type"], inplace=True)
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/agaricus-lepiota-tuning.csv')
        df_validation = read_csv('../data/agaricus-lepiota-validation.csv')
        df_train = read_csv('../data/agaricus-lepiota-train.csv')
        df_test = read_csv('../data/agaricus-lepiota-test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        columns = [ "poisonous",
        "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color",
        "population", "habitat"]
        df_tuning.columns = columns
        df_validation.columns = columns
        df_train.columns = columns
        df_test.columns = columns
        # Remove veil-type column in the datasets (only one value)
        if "veil-type" in df_tuning.columns:
            df_tuning.drop(columns=["veil-type"], inplace=True)
        if "veil-type" in df_validation.columns:
            df_validation.drop(columns=["veil-type"], inplace=True)
        if "veil-type" in df_train.columns:
            df_train.drop(columns=["veil-type"], inplace=True)
        if "veil-type" in df_test.columns:
            df_test.drop(columns=["veil-type"], inplace=True)
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/agaricus-lepiota.data', header = None)
    df.columns = [ "poisonous",
        "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color",
        "population", "habitat"]
    df = df.astype("str")
    # Remove veil-type column (only one value)
    df.drop(columns=["veil-type"], inplace=True)
    return df

def get_vancomycin_data():
    if params["split"]:
        df_train = read_csv('../data/vancomycin_train.csv')
        df_test = read_csv('../data/vancomycin_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/vancomycin_tuning.csv')
        df_validation = read_csv('../data/vancomycin_validation.csv')
        df_train = read_csv('../data/vancomycin_train.csv')
        df_test = read_csv('../data/vancomycin_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_tuning, df_validation, df_train, df_test
    if params["instances"] is None:
        df = read_csv('../data/vancomycin.csv')
    else:
        df = read_csv(f'../data/mimic-iii-preprocessed-db-sample-{params["instances"]}.csv')
    df = df.astype("str")
    df.columns = df.columns.astype(str)
    cols = df.columns.to_list()
    cols.remove("culture_susceptibility")
    cols.insert(0, "culture_susceptibility")
    df = df[cols]
    if not params["split"]:
        return df
    # stratified split
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["culture_susceptibility"], random_state=42)
    return df_train, df_test

def get_chess_data():
    if params["split"]:
        df_train = read_csv('../data/chess_train.csv')
        df_test = read_csv('../data/chess_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/chess-tuning.csv')
        df_validation = read_csv('../data/chess-validation.csv')
        df_train = read_csv('../data/chess_train.csv')
        df_test = read_csv('../data/chess_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/chess.csv', header = None)
    df.columns = ["bkblk","bknwy","bkon8","bkona","bkspr","bkxbq","bkxcr","bkxwp","blxwp","bxqsq","cntxt","dsopp","dwipd","hdchk","katri","mulch","qxmsq","r2ar8","reskd","reskr","rimmx","rkxwp","rxmsq","simpl","skach","skewr","skrxp","spcop","stlmt","thrsk","wkcti","wkna8","wknck","wkovl","wkpos","wtoeg","target"]
    # Target class as the first column
    cols = df.columns.to_list()
    cols.remove("target")
    cols.insert(0, "target")
    df = df[cols]
    df = df.astype("str")
    return df

def get_soybean_data():
    if params["split"]:
        df_train = read_csv('../data/soybean_train.csv')
        df_test = read_csv('../data/soybean_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/soybean-tuning.csv')
        df_validation = read_csv('../data/soybean-validation.csv')
        df_train = read_csv('../data/soybean_train.csv')
        df_test = read_csv('../data/soybean_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/soybean.csv', header = None)
    df.columns = ["class", "date", "plant-stand", "precip", "temp", "hail", "crop-hist", "area-damaged", "severity", "seed-tmt", "germination", "plant-growth", "leaves", "leafspots-halo", "leafspots-marg", "leafspot-size", "leaf-shread", "leaf-malf", "leaf-mild", "stem", "lodging", "stem-cankers", "canker-lesion", "fruiting-bodies", "external-decay", "mycelium", "int-discolor", "sclerotia", "fruit-pods", "fruit-spots", "seed", "mold-growth", "seed-discolor", "seed-size", "shriveling", "roots"]
    df = df.astype("str")
    return df

def get_healthy_aging_data():
    if params["split"]:
        df_train = read_csv('../data/healthy_aging_train.csv')
        df_test = read_csv('../data/healthy_aging_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/healthy_aging-tuning.csv')
        df_validation = read_csv('../data/healthy_aging-validation.csv')
        df_train = read_csv('../data/healthy_aging_train.csv')
        df_test = read_csv('../data/healthy_aging_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/healthy_aging.csv')
    df = df.astype("str")
    return df

def get_sick_data():
    if params["split"]:
        raise ValueError("Split not implemented for Sick database")
    if params["validation"]:
        raise ValueError("Validation not implemented for Sick database")
    df = datasets.load_sick_csv()
    df = df.astype("str")
    # Move target column to the first position
    df = df[[df.columns.to_list()[-1]] + df.columns.to_list()[:-1]]
    return df

def get_car_data():
    df = datasets.load_car_evaluation_csv()
    df = df.astype("str")
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=42)
    df_tuning, df_validation = train_test_split(df_train, test_size=0.2, stratify=df_train["class"], random_state=42)
    if params["split"]:
        return df_train, df_test
    if params["validation"]:
        return df_tuning, df_validation, df_train, df_test
    # Move target column to the first position
    df = df[[df.columns.to_list()[-1]] + df.columns.to_list()[:-1]]
    return df

def get_credit_data():
    if params["split"]:
        raise ValueError("Split not implemented for Credit database")
    if params["validation"]:
        raise ValueError("Validation not implemented for Credit database")
    df = datasets.load_credit_g_csv()
    df = df.astype("str")
    # Move target column to the first position
    df = df[[df.columns.to_list()[-1]] + df.columns.to_list()[:-1]]
    return df

def get_connect4_data():
    if params["split"]:
        df_train = read_csv('../data/connect-4_train.csv')
        df_test = read_csv('../data/connect-4_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/connect-4-tuning.csv')
        df_validation = read_csv('../data/connect-4-validation.csv')
        df_train = read_csv('../data/connect-4_train.csv')
        df_test = read_csv('../data/connect-4_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/connect-4.csv', header = None)
    df = df.astype("str")
    # Move target column to the first position
    df = df[[df.columns.to_list()[-1]] + df.columns.to_list()[:-1]]
    # Rename first column to "class"
    df.columns = ["class"] + [str(i) for i in df.columns.to_list()[1:]]
    return df

def get_student_success_data():
    if params["split"]:
        df_train = read_csv('../data/student_train.csv')
        df_test = read_csv('../data/student_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/student-tuning.csv')
        df_validation = read_csv('../data/student-validation.csv')
        df_train = read_csv('../data/student_train.csv')
        df_test = read_csv('../data/student_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_tuning, df_validation, df_train, df_test 
    df = read_csv('../data/student.csv')
    df = df.astype("str")
    return df

def get_dermatology_data():
    if params["split"]:
        df_train = read_csv('../data/dermatology_train.csv')
        df_test = read_csv('../data/dermatology_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/dermatology_tuning.csv')
        df_validation = read_csv('../data/dermatology_validation.csv')
        df_train = read_csv('../data/dermatology_train.csv')
        df_test = read_csv('../data/dermatology_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/dermatology.csv')
    df = df.astype("str")
    return df

def get_depression_data():
    if params["split"]:
        df_train = read_csv('../data/depression_train.csv')
        df_test = read_csv('../data/depression_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/depression_tuning.csv')
        df_validation = read_csv('../data/depression_validation.csv')
        df_train = read_csv('../data/depression_train.csv')
        df_test = read_csv('../data/depression_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/depression.csv')
    df = df.astype("str")
    return df

def get_mimic_iv_data():
    if params["split"]:
        df_train = read_csv('../data/mimic-iv-staph-train.csv')
        df_test = read_csv('../data/mimic-iv-staph-test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        df_train.columns = df_train.columns.astype(str)
        df_test.columns = df_test.columns.astype(str)
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/mimic-iv-staph-tuning.csv')
        df_validation = read_csv('../data/mimic-iv-staph-validation.csv')
        df_train = read_csv('../data/mimic-iv-staph-train.csv')
        df_test = read_csv('../data/mimic-iv-staph-test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        df_tuning.columns = df_tuning.columns.astype(str)
        df_validation.columns = df_validation.columns.astype(str)
        df_train.columns = df_train.columns.astype(str)
        df_test.columns = df_test.columns.astype(str)
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/mimic-iv-staph.csv')
    df = df.astype("str")
    df.columns = df.columns.astype(str)
    return df

def get_cardio_data():
    if params["split"]:
        df_train = read_csv('../data/cardio_train.csv')
        df_test = read_csv('../data/cardio_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        df_train.columns = df_train.columns.astype(str)
        df_test.columns = df_test.columns.astype(str)
        return df_train, df_test
    if params["validation"]:
        df_tuning = read_csv('../data/cardio_tuning.csv')
        df_validation = read_csv('../data/cardio_validation.csv')
        df_train = read_csv('../data/cardio_train.csv')
        df_test = read_csv('../data/cardio_test.csv')
        df_tuning = df_tuning.astype("str")
        df_validation = df_validation.astype("str")
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        df_tuning.columns = df_tuning.columns.astype(str)
        df_validation.columns = df_validation.columns.astype(str)
        df_train.columns = df_train.columns.astype(str)
        df_test.columns = df_test.columns.astype(str)
        return df_tuning, df_validation, df_train, df_test
    df = read_csv('../data/cardio.csv')
    df = df.astype("str")
    df.columns = df.columns.astype(str)
    return df

def get_facebook_data():
    if params["split"]:
        df_train = read_csv('../data/facebook_train.csv')
        df_test = read_csv('../data/facebook_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    df = read_csv('../data/facebook.csv')
    df = df.astype("str")
    return df

def get_synthetic_data():
    if params["split"]:
        df_train = read_csv('../data/synthetic3_train.csv')
        df_test = read_csv('../data/synthetic3_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    df = read_csv('../data/synthetic3_all.csv')
    df = df.astype("str")
    return df

def get_polluted_synthetic_data():
    if params["split"]:
        df_train = read_csv('../data/synthetic3-5_train.csv')
        df_test = read_csv('../data/synthetic3-5_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    df = read_csv('../data/synthetic3-5_all.csv')
    df = df.astype("str")
    return df

def get_brca_n_data():
    if params["split"]:
        df_train = read_csv('../data/brca_n_train.csv')
        df_test = read_csv('../data/brca_n_test.csv')
        df_train = df_train.astype("str")
        df_test = df_test.astype("str")
        return df_train, df_test
    df = read_csv('../data/brca_n.csv')
    df = df.astype("str")
    return df


def get_target(database:str):
    if database == "MIMIC":
        return ("interpretation", "R")
    elif database == "Mushrooms":
        return ("poisonous", "e")
    elif database == "Vancomycin":
        return ("culture_susceptibility", "R")
    elif database == "Chess":
        return ("target", "won")
    elif database == "Soybean":
        return ("class", "alternarialeaf-spot")
    elif database == "HealthyAging":
        return ("Number_of_Doctors_Visited", "1")
    elif database == "sick":
        return ("class", "sick")
    elif database == "credit":
        return ("class", "bad")
    elif database == "connect4":
        return ("class", "win")
    elif database == "StudentSuccess":
        return ("Target", "Graduate")
    elif database == "Car":
        return ("class", "unacc")
    elif database == "Dermatology":
        return ("class", "1")
    elif database == "Depression":
        return ("Depression", "Yes")
    elif database == "MIMIC-IV":
        return ("interpretation", "R")
    elif database == "Cardio":
        return ("cardio", "1")
    elif database == "Facebook":
        return ("target", "government")
    elif database == "synthetic":
        return ("Target", "1")
    elif database == "polluted_synthetic":
        return ("Target", "1")
    elif database == "brca_n":
        return ("Target", "1")
    else:
        raise ValueError("Database not recognized")
    
    
def output_csv(model, id, df = None, target = None, database = None):
    if database is not None:
        input_file = f"results_{model}_{id}_{database}.txt"
    else:
        input_file = f"results_{model}_{id}.txt"
    is_quality = model in quality_models
    if not is_quality and (df is None or target is None):
        raise ValueError("Need to pass df if model is not quality based")
    if is_quality:
        output_df = DataFrame(columns=["Description","Quality","tp","fp", "TP","FP"])
    else:
        output_df = DataFrame(columns=["Description","tp","fp", "TP","FP"])
        target_entry = df[target[0]] == target[1]
    # with open(input_file, "r") as f:
    try:
        f = open(input_file, "r")
    except Exception:
        print(f"WARNING: No results found for {model} with id {id}")
        return output_df
    # Description: [ab_name = 'LEVOFLOXACIN', exitus = '0'], Target: interpretation = 'R' ; Rank : 5 ; coverage : 0.10392986828817602 ; odds_ratio : 2.5780732920078053 ; p_value : 0.0 ; absolute_contribution : 0.8948480217833793 ; contribution_ratio : 3.2350923761466572 ; 
    # Description: [days_with_treatment_last_180_days = '0', gender = 'M'], Target: interpretation = 'R' ; Quality Measure WRAcc = 0.006993814700406782 ; tp = 34406 ; fp = 78977 ; TP = 61727 ; FP = 150879
    for line in f:
        description_str = line.split("Description: [")[1].split("], Target:")[0]
        if is_quality:
            quality_measure_name = params["quality_measure"].get_name()
            quality = float(line.split(f"Quality Measure {quality_measure_name} = ")[1].split(" ;")[0])
            tp = int(line.split("tp = ")[1].split(" ;")[0])
            fp = int(line.split("fp = ")[1].split(" ;")[0])
            TP = int(line.split("TP = ")[1].split(" ;")[0])
            FP = int(line.split("FP = ")[1].split(" ;")[0])
            output_df.loc[len(output_df)] = [description_str, quality, tp, fp, TP, FP]
        else:
            entry = Series(True, index=df.index)
            for selector in description_str.split(", "):
                column = selector.split(" = ")[0]
                value = selector.split(" = ")[1]
                value = value.replace("'", "")
                entry = entry & (df[column] == value)
            tp = sum(entry & target_entry)
            fp = sum(entry & ~target_entry)
            TP = sum(target_entry)
            FP = len(target_entry) - TP
            output_df.loc[len(output_df)] = [description_str, tp, fp, TP, FP]
    # os.remove(input_file)
    if params["num_subgroups"] is not None and "Quality" in output_df.columns:
        output_df.sort_values(by="Quality", ascending=False, inplace=True)
        output_df = output_df.iloc[0:params["num_subgroups"]]
    f.close()
    return output_df
    
def output_csv_test(train_output, df_test, target):
    output_df = train_output.copy()
    target_entry = df_test[target[0]] == target[1]
    for i in range(len(output_df)):
        description_str = output_df.iloc[i]
        description_str = description_str.Description
        entry = Series(True, index=df_test.index)
        for selector in description_str.split(", "):
            column = selector.split(" = ")[0]
            value = selector.split(" = ")[1]
            value = value.replace("'", "")
            entry = entry & (df_test[column] == value)
        tp = sum(entry & target_entry)
        fp = sum(entry & ~target_entry)
        TP = sum(target_entry)
        FP = len(target_entry) - TP
        output_df.loc[output_df.index[i], "tp test"] = tp
        output_df.loc[output_df.index[i], "fp test"] = fp
        output_df.loc[output_df.index[i], "TP test"] = TP
        output_df.loc[output_df.index[i], "FP test"] = FP
    output_df.rename(columns={"tp":"tp train","fp":"fp train","TP":"TP train","FP":"FP train"}, inplace=True)
    return output_df

def compute_covered_instances(model, id, df, target, database = None, remove_file = True, or_val = None, ppv_val = None):
    if database is not None and or_val is not None and ppv_val is not None:
        input_file = f"results_{model}_{id}_{database}_{or_val}-{ppv_val}.txt"
    elif database is not None:
        input_file = f"results_{model}_{id}_{database}.txt"
    else:
        input_file = f"results_{model}_{id}.txt"
    covered_instances = Series(0, index=df.index)
    try:
        f = open(input_file, "r")
    except Exception:
        print(f"WARNING: No results found for {model} with id {id}")
        print("Input file:", input_file)
        return covered_instances
    # Description: [ab_name = 'LEVOFLOXACIN', exitus = '0'], Target: interpretation = 'R' ; Rank : 5 ; coverage : 0.10392986828817602 ; odds_ratio : 2.5780732920078053 ; p_value : 0.0 ; absolute_contribution : 0.8948480217833793 ; contribution_ratio : 3.2350923761466572 ; 
    # Description: [days_with_treatment_last_180_days = '0', gender = 'M'], Target: interpretation = 'R' ; Quality Measure WRAcc = 0.006993814700406782 ; tp = 34406 ; fp = 78977 ; TP = 61727 ; FP = 150879
    count = 0
    for line in f:
        description_str = line.split("Description: [")[1].split("], Target:")[0]
        entry = Series(True, index=df.index)
        for selector in description_str.split(", "):
            column = selector.split(" = ")[0]
            value = selector.split(" = ")[1]
            value = value.replace("'", "")
            entry = entry & (df[column] == value)
        covered_instances = covered_instances | entry
        count += 1
        # If model is SDMapStar, only consider the first param["num_sugropus"] subgroups
        if model == "SDMapStar" and count == params["num_subgroups"]:
            break
    # If model is SDMapStar, ensure that the number of subgroups is params["num_subgroups"]
    if model == "SDMapStar" and count < params["num_subgroups"]:
        raise ValueError("Number of subgroups is less than params['num_subgroups']",count, params["num_subgroups"],". Change thresholds for this dataset for k", params["num_subgroups"] ," with columns", len(df.columns),"\n The columns are:", df.columns)
    target_entry = df[target[0]] == target[1]
    total_covered_instances = covered_instances.sum() / len(df)
    target_covered_instances = sum(covered_instances & target_entry) / sum(target_entry)
    negative_target_covered_instances = sum(covered_instances & ~target_entry) / sum(~target_entry)
    f.close()
    if remove_file:
        os.remove(input_file)
    return total_covered_instances, target_covered_instances, negative_target_covered_instances

def benchmark(model, database):
    if params["columns"] is not None and params["columns"] < params["max_size"]:
        raise ValueError("Columns must be at least " + str(params["max_size"]))
    df = get_data(database)
    if params["columns"] is not None:
        df = df.iloc[:,0:params["columns"]+1] # +1 for target
    target = get_target(database)
    model, id = get_model(model, write = False)
    t0 = time.process_time()
    model.fit(df, target)
    t1 = time.process_time()
    p = psutil.Process()
    mem = p.memory_info().rss
    return t1-t0, mem

def run(model, database):
    if params["split"]:
        df_train, df_test = get_data(database)
        target = get_target(database)
        sd, id = get_model(model, write = True, database=database)
        sd.fit(df_train, target)
        output_train = output_csv(model, id, df_train, target, database)
        output_test = output_csv_test(output_train, df_test, target)
        return output_test
    else:
        df = get_data(database)
        target = get_target(database)
        sd, id = get_model(model, write = True, database=database)
        sd.fit(df, target)
        return output_csv(model, id, df, target, database)
    
def run_cross_validation(model, database):
    df = get_data(database)
    target = get_target(database)
    folds = get_folds(df,target)
    output_df = DataFrame()
    for i in range(len(folds)):
        train, test = folds[i]
        sd, id = get_model(model, write = True, database=database)
        sd.fit(train, target)
        output_train = output_csv(model, id, train, target, database)
        output_test = output_csv_test(output_train, test, target)
        output_df = concat([output_df, output_test])
    return output_df

def write_subgroups(model, database):
    df = get_data(database)
    target = get_target(database)
    sd, id = get_model(model, write = True, database=database)
    sd.fit(df, target)

def covered_instances(model, database):
    df = get_data(database)
    target = get_target(database)
    sd, id = get_model(model, write = True, database=database)
    sd.fit(df, target)
    return compute_covered_instances(model, id, df, target, database)


def split_covered_instances(model, database):
    params["split"] = True
    df_train, df_test = get_data(database)
    target = get_target(database)
    sd, id = get_model(model, write = True, database=database)
    sd.fit(df_train, target)
    total_covered_instances_train, target_covered_instances_train, negative_target_covered_instances_train = compute_covered_instances(model, id, df_train, target, database, remove_file = False)
    total_covered_instances_test, target_covered_instances_test, negative_target_covered_instances_test = compute_covered_instances(model, id, df_test, target, database, remove_file=True)
    return total_covered_instances_train, target_covered_instances_train, negative_target_covered_instances_train, total_covered_instances_test, target_covered_instances_test, negative_target_covered_instances_test, sd.selected_subgroups

def validation_covered_instances(model, database):
    params["validation"] = True
    df_tuning, df_validation, df_train, df_test = get_data(database)
    target = get_target(database)
    sd, id = get_model(model, write = True, database=database)
    sd.fit(df_tuning, target)
    total_covered_instances_validation, target_covered_instances_validation, negative_target_covered_instances_validation = compute_covered_instances(model, id, df_validation, target, database, remove_file=True)
    return total_covered_instances_validation, target_covered_instances_validation, negative_target_covered_instances_validation, sd.selected_subgroups



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Main Experiments Script',
                    description= 'Run experiments for subgroups algorithms',
                    )
    
    # Mandatory arguments
    parser.add_argument('mode', type=str, help='Mode to run the script')
    parser.add_argument('model', type=str, help='Model to run the experiments')
    parser.add_argument('database', type=str, help='Database to run the experiments')
    # Optional arguments
    parser.add_argument('--instances', type=int, help='Number of instances to run the experiments')
    parser.add_argument('--columns', type=int, help='Number of columns to run the experiments')
    parser.add_argument('--split', type=bool, help='Whether to split the data or not')
    parser.add_argument('--num_subgroups', type=int, help='Number of subgroups to return')
    parser.add_argument('--max_size', type=int, help='Maximum size of the subgroups')
    parser.add_argument('--coverage_thld', type=float, help='Coverage threshold')
    parser.add_argument('--ppv_thld', type=float, help='PPV threshold')
    parser.add_argument('--or_thld', type=float, help='OR threshold')
    parser.add_argument('--p_val_thld', type=float, help='P-value threshold')
    parser.add_argument('--abs_contribution_thld', type=float, help='Absolute contribution threshold')
    parser.add_argument('--contribution_thld', type=float, help='Contribution threshold')
    parser.add_argument('--min_rank', type=int, help='Minimum rank')
    parser.add_argument('--folds', type=int, help='Number of folds for cross validation')
    parser.add_argument('--quality_measure', type=str, help='Quality measure to use. Default is WRAcc')
    args = parser.parse_args()
    mode = args.mode
    model = args.model
    database = args.database
    if args.instances is not None:
        params["instances"] = args.instances
    if args.columns is not None:
        params["columns"] = args.columns
    if args.split is not None:
        params["split"] = args.split
    if args.num_subgroups is not None:
        params["num_subgroups"] = args.num_subgroups
    if args.max_size is not None:
        params["max_size"] = args.max_size
    if args.coverage_thld is not None:
        params["coverage_thld"] = args.coverage_thld
    if args.ppv_thld is not None:
        params["ppv_thld"] = args.ppv_thld
    if args.or_thld is not None:
        params["or_thld"] = args.or_thld
    if args.p_val_thld is not None:
        params["p_val_thld"] = args.p_val_thld
    if args.abs_contribution_thld is not None:
        params["abs_contribution_thld"] = args.abs_contribution_thld
    if args.contribution_thld is not None:
        params["contribution_thld"] = args.contribution_thld
    if args.min_rank is not None:
        params["min_rank"] = args.min_rank
    if args.folds is not None:
        params["folds"] = args.folds
    if args.quality_measure is not None:
        if args.quality_measure == "WRAcc":
            params["quality_measure"] = WRAcc()
            params["optimistic_estimate"] = WRAccOptimisticEstimate1()
        elif args.quality_measure == "BinomialTest":
            params["quality_measure"] = BinomialTest()
            params["optimistic_estimate"] = BinomialTestOptimisticEstimate1()
        elif args.quality_measure == "PiatetskyShapiro":
            params["quality_measure"] = PiatetskyShapiro()
            params["optimistic_estimate"] = PiatetskyShapiroOptimisticEstimate1()
        else:
            raise ValueError("Quality measure not recognized")

    if mode == "benchmark":
        print(benchmark(model,database))
    elif mode == "run":
        output_df = run(model,database)
        if params["columns"] is not None:
            output_df.to_csv(f"results_{model}_{database}_{params['columns']}.csv")
        else:
            output_df.to_csv(f"results_{model}_{database}.csv")
    elif mode == "cross_validation":
        output_df = run_cross_validation(model,database)
        output_df.to_csv(f"results_{model}_{database}_cross_validation.csv")
    elif mode == "write_subgroups":
        write_subgroups(model,database)
    elif mode == "covered_instances":
        total_covered_instances, target_covered_instances, negative_target_covered_instances = covered_instances(model,database)
        print("Total covered instances: ", total_covered_instances)
        print("Target covered instances: ", target_covered_instances)
        print("Negative target covered instances: ", negative_target_covered_instances)
    elif mode == "split_covered_instances":
        total_covered_instances_train, target_covered_instances_train, negative_target_covered_instances_train, total_covered_instances_test, target_covered_instances_test, negative_target_covered_instances_test, selected_subgroups = split_covered_instances(model,database)
        print("Total covered instances train: ", total_covered_instances_train)
        print("Target covered instances train: ", target_covered_instances_train)
        print("Negative target covered instances train: ", negative_target_covered_instances_train)
        print("Total covered instances test: ", total_covered_instances_test)
        print("Target covered instances test: ", target_covered_instances_test)
        print("Negative target covered instances test: ", negative_target_covered_instances_test)
        print("Selected subgroups: ", selected_subgroups)
    elif mode == "validation_covered_instances":
        total_covered_instances_validation, target_covered_instances_validation, negative_target_covered_instances_validation, selected_subgroups = validation_covered_instances(model,database)
        print("Total covered instances validation: ", total_covered_instances_validation)
        print("Target covered instances validation: ", target_covered_instances_validation)
        print("Negative target covered instances validation: ", negative_target_covered_instances_validation)
        print("Selected subgroups: ", selected_subgroups)
    elif mode == "print_params":
        for key in params:
            print(key, ":", params[key])
    else:
        print("Mode not recognized")