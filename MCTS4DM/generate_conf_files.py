from itertools import product


grid_parameters = {
    "redundancyStrategy": ["jaccardSupportDescription", "jaccardSupportDescriptionTarget", "sumJaccard"],
    "maxRedundancy": [0.4, 0.6, 0.8],
    "nbIter" : [1000, 10000, 100000],
    "measure" : ["WRAcc", "WeightedRelativeF1", "RelativeF1", "Acc"],
    "attrFile": ["datasets/brca_n/properties_train.csv"],
    "targetFile": ["datasets/brca_n/qualities_train.csv"],
    "resultFolderName": ["brcaN"]

}

parameters_template = open("parameters.conf", "r")

lines = parameters_template.readlines()

for redundancyStrategy, maxRedundancy, nbIter, measure, attrFile, targetFile, resultFolderName in product(*grid_parameters.values()):
    new_lines = []
    resultFolderName = f"{resultFolderName}_{redundancyStrategy}_{maxRedundancy}_{nbIter}_{measure}"
    for line in lines:
        if line.startswith("redundancyStrategy"):
            new_lines.append(f"redundancyStrategy = {redundancyStrategy}\n")
        elif line.startswith("maxRedundancy"):
            new_lines.append(f"maxRedundancy = {maxRedundancy}\n")
        elif line.startswith("nbIter"):
            new_lines.append(f"nbIter = {nbIter}\n")
        elif line.startswith("measure"):
            new_lines.append(f"measure = {measure}\n")
        elif line.startswith("attrFile"):
            new_lines.append(f"attrFile = {attrFile}\n")
        elif line.startswith("targetFile"):
            new_lines.append(f"targetFile = {targetFile}\n")
        elif line.startswith("resultFolderName"):
            new_lines.append(f"resultFolderName = {resultFolderName}\n")
        else:
            new_lines.append(line)
    conf_name = f"parameters_{redundancyStrategy}_{maxRedundancy}_{nbIter}_{measure}.conf"
    with open("ConfigurationFiles/"+conf_name, "w+") as f:
        f.writelines(new_lines)
