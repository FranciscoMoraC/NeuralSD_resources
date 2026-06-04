from neuralsd.quality_measure import QualityMeasure
import torch

class WRAcc(QualityMeasure):
    """
    Class for the WRAcc quality measure.
    Inherits from the QualityMeasure abstract class.
    """

    def compute_quality(self, tp, fp, TP, FP, additional_params_for_quality_measure=dict()):
        """
        Compute the WRAcc quality measure.

        Args:
            tp (int): True positives.
            fp (int): False positives.
            TP (int): Total positives.
            FP (int): Total false positives.
            additional_params_for_quality_measure (dict, optional): Additional parameters for the quality measure.

        Returns:
            float: The computed WRAcc quality measure.
        """
        N = TP + FP
        return (tp+fp)/N*(tp/(tp+fp) - TP/N)

    def gradient(self, g, target, TP, FP, k, additional_params_for_quality_measure=dict()):
        """
        Compute the gradient of the WRAcc quality measure.

        Args:
            g: Degree of coverage.
            target: Target class vector.
            TP (int): Total positives.
            FP (int): Total false positives.
            k (int): Number of subgroups to consider.
            additional_params_for_quality_measure (dict, optional): Additional parameters for the quality measure.
        Returns:
            tensort: The gradient of the WRAcc quality measure with respect to g.
        """        
        N = TP + FP

        dL_dg = -1/N * (target - TP/N) # shape: (1,n)
        dL_dg = dL_dg.repeat(k, 1)
        dL_dg = dL_dg.to(torch.double)

        return dL_dg