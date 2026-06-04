import torch
from pandas import DataFrame, Series
import time
from selector_matrix import SelectorMatrix
from nsd import NeuralSD

class IterativeNSD(NeuralSD):
    """
    Iterative Neural Subgroup Discovery (NSD) class.
    This class extends the NeuralSD class to perform the iterative methodology for subgroup discovery.
    """

    def __init__(self, k, quality_measure, additional_params_for_quality_measure=dict()):
        """
        Initialize the IterativeNSD class.

        Args:
            k (int): The number of subgroups to discover.
            quality_measure (callable): The quality measure function.
            diversity (bool): Whether to use diversity in the quality measure.
            additional_params_for_quality_measure (dict): Additional parameters for the quality measure.
        """
        super().__init__(k, quality_measure, additional_params_for_quality_measure)
        self.subgroups = []

    def fit(self, df: DataFrame, target_as_tuple: tuple, max_iterations, min_TP = None, min_quality = None, weight_init="random",
            additional_params_for_weight_init=dict(),epochs = 1000, lr=100, patience=2, verbose = True, threshold_for_subgroup_description = 0.5,
            batch_size=None):
        if type(max_iterations) is not int:
            raise TypeError("max_iterations must be an integer")
        if max_iterations < 1:
            raise ValueError("max_iterations must be greater than 0")
        if min_TP is not None and type(min_TP) is not int:
            raise TypeError("min_TP must be an integer")
        if min_TP is not None and min_TP < 1:
            raise ValueError("min_TP must be greater than 0")
        if min_quality is not None and type(min_quality) is not float and type(min_quality) is not int:
            raise TypeError("min_quality must be a float or an integer")
        if type(df) != DataFrame:
            raise TypeError("df must be a pandas DataFrame")
        # Since we are removing instances from the DataFrame, we need to make a copy of it
        df = df.copy()
        history = []
        iteration = 1
        TP = (df[target_as_tuple[0]] == target_as_tuple[1]).sum()
        t0 = time.time()
        while iteration <= max_iterations and (min_TP is None or TP >= min_TP) and TP > 0:
            if verbose:
                print(f"Iteration {iteration} of {max_iterations}")
                print(f"TP: {TP}")
            qualities = super().fit(df, target_as_tuple, weight_init=weight_init, additional_params_for_weight_init=additional_params_for_weight_init,
                        epochs=epochs, lr=lr, patience=patience, verbose=verbose, batch_size=batch_size)
            # Index of the best subgroup
            best_subgroup = torch.argmax(qualities)
            # If the minimum quality is set and the best subgroup does not meet it, break the loop
            if min_quality is not None and qualities[best_subgroup] < min_quality:
                if verbose:
                    print(f"Best subgroup quality {qualities[best_subgroup]} is less than minimum quality {min_quality}. Stopping.")
                break
            history.append(self.quality_records)
            # Get the subgroup description and save it in the list of discovered subgroups
            best_subgroup_description = super().get_subgroup_descriptions(threshold_for_subgroup_description)[best_subgroup]
            quality, tp, fp = self.forward(report_tp_fp=True)
            self.subgroups.append((best_subgroup_description, quality[best_subgroup].item(), tp[best_subgroup].item(), fp[best_subgroup].item()))
            if verbose:
                print(f"Best subgroup description: {best_subgroup_description}")
                print(f"Quality: {quality[best_subgroup].item()}, tp: {tp[best_subgroup].item()}, fp: {fp[best_subgroup].item()}")
            # Remove the instances covered by the best subgroup from the DataFrame
            instances_covered = Series(True, index=df.index)
            for col,val in best_subgroup_description:
                instances_covered = instances_covered & (df[col] == val) 
            df = df[~instances_covered]
            TP = (df[target_as_tuple[0]] == target_as_tuple[1]).sum()
            iteration += 1
        if verbose:
            print(f"Finished in {iteration-1} iterations")
            print(f"Time taken: {time.time() - t0:.2f} seconds")
        return history
    
    def get_subgroup_descriptions(self):
        """
        Get the descriptions of the discovered subgroups.

        Returns:
            list: A list of subgroup descriptions.
        """
        return self.subgroups