# Gradient Descent Subgroup Discovery without Dimensionality Reduction for Biomedical Data

## Algorithm and experiment code

The code for the NeuralSD algorithm can be found in the `neuralsd` folder.

The code for the experiments is split between each method applied.

We included reference for the data used in the main paper. All data but the Vancomycin dataset is available online. For the Vancomycin dataset, which was extracted from the MIMIC database, we reference the main work which provided the SQL queries.

The `data_generator` folder contains the code for generating the synthetic datasets used in the experiments, as well as the input file for generating the synthetic datasets with the desired structure (3 pure subgroups of varying number of selectors and samples).

The `main.py` and `main_sdiga.py` files include code for running BSD and SDIGA and importing the data for both experiments.

A `requierements.txt` file is included with the used libraries and their versions. We recommend using a virtual environment to install the dependencies.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

