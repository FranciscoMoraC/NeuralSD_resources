from pandas import DataFrame
import torch

class SelectorMatrix:
    def __init__(self, df: DataFrame, target_as_tuple : tuple):
        """
        Initialize the SelectorMatrix with a DataFrame, leaving out the target variable.

        Args:
            df (DataFrame): The DataFrame containing the data.
            target_as_tuple (tuple): A tuple representing the target variable (e.g., ('target', 'value')).
        """
        self._selector_descriptions = []
        for c in df.columns:
            if c != target_as_tuple[0]:
                for v in df[c].unique():
                    self._selector_descriptions.append((c, v))
        self._matrix = torch.zeros((len(self._selector_descriptions), len(df)))
        for i, (c, v) in enumerate(self._selector_descriptions):
            self._matrix[i] = torch.tensor((df[c] == v).to_numpy())

    def _get_selector_matrix(self):
        """
        Get the selector matrix.

        Returns:
            torch.Tensor: The selector matrix.
        """
        return self._matrix
    
    def _set_selector_matrix(self, matrix: torch.Tensor):
        """
        Set the selector matrix.

        Args:
            matrix (torch.Tensor): The new selector matrix.
        """
        self._matrix = matrix

    def _get_selector_descriptions(self):
        """
        Get the selector descriptions.
        Returns:
            list: The selector descriptions.
        """
        return self._selector_descriptions
    
    matrix = property(_get_selector_matrix, _set_selector_matrix)
    selector_descriptions = property(_get_selector_descriptions)

    def get_subgroup_description(self, subgroup: torch.Tensor):
        """
        Get the subgroup description for a given subgroup.

        Args:
            subgroup (torch.Tensor): The subgroup represented as a tensor of size (1,n_selectors).
            The tensor should contain 1s and 0s, where 1 indicates the presence of a selector.

        Returns:
            list: A list of selector descriptions for the selected subgroup.
        """
        return [self._selector_descriptions[i] for i in range(len(subgroup)) if subgroup[i]]      