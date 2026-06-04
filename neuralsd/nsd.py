import torch
from pandas import DataFrame
import time
from selector_matrix import SelectorMatrix

##TODO: Change measures code to match the library when adding the algorithm to the library

class NeuralSD:

    available_quality_measures = ["PPV", "WRAcc", "Qc", "Qg"]

    def _get_weight_init_methods(self):
        return {
            "random": self._random_init,
            "best": self._best_selectors_init,
            "manual": self._manual_init
        }

    def __init__(self, k, quality_measure, additional_params_for_quality_measure=dict()):
        self._k = k
        if quality_measure not in self.available_quality_measures:
            raise ValueError("Quality measure must be one of {}".format(self.available_quality_measures))
        self._additional_params_for_quality_measure = additional_params_for_quality_measure
        self.quality_measure = quality_measure

    def _create_target_tensor(self,df,target_tuple):
        target = df[target_tuple[0]] == target_tuple[1]
        target = target.to_numpy().astype(int)
        target = target.reshape((1, -1))
        target = torch.tensor(target)
        return target

    def fit(self, df: DataFrame, target_as_tuple: tuple, weight_init="random", additional_params_for_weight_init=dict(), epochs = 1000, lr=100, patience=2, verbose = True, batch_size = None):
        """
        Fit the model to the data.

        :param df: A DataFrame containing the data.
        :param target_as_tuple: A tuple representing the target variable (e.g., ('target', 'value')).
        :param weight_init: The method for initializing weights. Default is "random".
        :param additional_params_for_weight_init: Additional parameters for weight initialization.
        """
        if not isinstance(df, DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(target_as_tuple, tuple):
            raise ValueError("target_as_tuple must be a tuple")
        if len(target_as_tuple) != 2:
            raise ValueError("target_as_tuple must be of length 2")
        if target_as_tuple[0] not in df.columns:
            raise ValueError("target_as_tuple[0] must be a column in df")
        if target_as_tuple[1] not in df[target_as_tuple[0]].unique():
            raise ValueError("target_as_tuple[1] must be a value in df[target_as_tuple[0]]")
        weight_init_methods = self._get_weight_init_methods()
        if weight_init not in weight_init_methods:
            raise ValueError("weight_init must be one of {}".format(list(self._get_weight_init_methods().keys())))
        # Drop last rows if batch_size is not None and df is not divisible by batch_size
        if batch_size is not None:
            if not isinstance(batch_size, int):
                raise ValueError("batch_size must be an integer")
            if batch_size < 1:
                raise ValueError("batch_size must be greater than 0")
            if df.shape[0] % batch_size != 0:
                df = df.iloc[:-(df.shape[0] % batch_size)]
        # Initialize the selector matrix
        self._selector_matrix = SelectorMatrix(df, target_as_tuple)
        self._selectors = self._selector_matrix.matrix.shape[0]
        self._rows = self._selector_matrix.matrix.shape[1]
        # Initialize the target tensor
        self.target = self._create_target_tensor(df, target_as_tuple)
        self.TP = self.target.sum()
        self.N = self._rows
        self.FP = self.N - self.TP
        # Initialize W
        weight_init_methods[weight_init](additional_params_for_weight_init)
        quality = self.forward()
        curr_patience = patience
        # Matrices used to compute the quality in the entire dataset when using mini-batches
        if batch_size is not None:
            global_selector_matrix = self._selector_matrix.matrix.clone().detach()
            global_target = self.target.clone().detach()
        t0 = time.time()
        self.quality_records = []
        for epoch in range(1, epochs+1):
            # When not using mini-batches, the entire dataset is used in each epoch
            if batch_size is None:
                self.backward(lr)
                new_quality = self.forward()
            else:
                # Perform batch training
                for i in range(0, self._rows, batch_size):
                    batch_target = global_target[:, i:i+batch_size]
                    batch_selectors = global_selector_matrix[:, i:i+batch_size]
                    self._selector_matrix.matrix = batch_selectors
                    self.target = batch_target
                    # Adjust TP, FP, N for the batch
                    self.TP = self.target.sum()
                    self.N = batch_selectors.shape[1]
                    self.FP = self.N - self.TP
                    self.backward(lr)
                # Compute quality for the entire dataset
                self._selector_matrix.matrix = global_selector_matrix
                self.target = global_target
                self.TP = self.target.sum()
                self.N = global_selector_matrix.shape[1]
                self.FP = self.N - self.TP
                new_quality = self.forward()
            if (new_quality-quality < 1e-10).all():
                curr_patience -= 1
                if curr_patience == 0:
                    break
            else:
                curr_patience = patience
                quality = new_quality
            self.quality_records.append(quality.item())
            if epoch % 10 == 0 and verbose:
                print("Epoch", epoch)
                print("Time taken: ", time.time() - t0)
                print("Quality", quality)
                print("Descriptions:", self.get_subgroup_descriptions())
        if verbose:
            print("Epochs", epoch)
            print("Time taken: ", time.time() - t0)
            print("Quality", quality)
        return quality        

    """Initialize each subgroup using a random uniform distribution.

    :param additional_parameters: A dictionary. It may contain the following keys:
        - distribution: Either "uniform" or "normal". Default is "uniform".
    """
    def _random_init(self, additional_parameters = dict()):
        if "distribution" in additional_parameters:
            if additional_parameters["distribution"] == "uniform":
                self.W = torch.rand(self._selectors, self._k, dtype=torch.float64)
            elif additional_parameters["distribution"] == "normal":
                self.W = torch.randn(self._selectors, self._k, dtype=torch.float64)
            else:
                raise ValueError("distribution must be either 'uniform' or 'normal'")
        else:
            self.W = torch.rand(self._selectors, self._k, dtype=torch.float64)

    """Initialize each subgroup using a provided tensor.
    
    :param additional_parameters: A dictionary containing the following keys:
        - weights: A tensor of shape (selectors, k) representing the initial weights.
    """
    def _manual_init(self, additional_parameters):
        if "weights" not in additional_parameters:
            raise ValueError("weights must be provided for manual initialization")
        weights = additional_parameters["weights"]
        if weights.shape != (self._selectors, self._k):
            raise ValueError("weights shape must be ({}, {})".format(self._selectors, self._k))
        self.W = weights

    """Initialize each subgroup using the best selectors based on the quality measure. Each subgroup is initialized with one of the top k selectors.
    
    :param additional_parameters: A dictionary. It does not need to contain any keys.
    """
    def _best_selectors_init(self, additional_parameters = dict()):
        x = self._selector_matrix.matrix
        # Compute the quality measure for each selector
        tp = (x * self.target).sum(dim=1)
        fp = (x * (1 - self.target)).sum(dim=1)
        if self.quality_measure == "PPV":
            quality = tp/(tp+fp)
        elif self.quality_measure == "WRAcc":
            quality = (tp+fp)/self.N*(tp/(tp+fp) - self.TP/self.N)
        elif self.quality_measure == "Qc":
            quality = tp-self._additional_params_for_quality_measure["c"]*fp
        elif self.quality_measure == "Qg":
            quality = tp / (fp + self._additional_params_for_quality_measure["g"])
        # Select the best k selectors based on the quality measure
        # If n is provided in additional_parameters, select the top n selectors, otherwise select the top k selectors
        if "n" in additional_parameters:
            best_indices = torch.topk(quality, additional_parameters["n"])[1]
        else:
            best_indices = torch.topk(quality, self._k)[1]
        # Initialize W with the k best selectors only
        self.W = torch.full((self._selectors, self._k), -10.0)
        if "n" in additional_parameters:
            for p in best_indices:
                self.W[p, :] = 0
        else:
            for idx, p in enumerate(best_indices):
                self.W[p, idx] = 10

    def forward(self, report_tp_fp=False):
        X = self._selector_matrix.matrix.unsqueeze(0).expand(self._k, -1, -1).transpose(0, 1)
        self.W1 = torch.sigmoid(self.W)
        # W is pxk, create 3-dimensional tensor of size k x n x p
        W1 = self.W1.unsqueeze(1).expand(-1, self.N, -1).mT
        # Fuzzy subgroup instances
        g = 1-W1*(1-X)
        self.g = g.prod(0)
        # Swap 1st and 2nd dimensions
        self.g = self.g.transpose(0, 1)
        tp = self.g * self.target.t()
        fp = self.g * (1 - self.target.t())
        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        quality = None
        if self.quality_measure == "PPV":
            quality = tp/(tp+fp)
        elif self.quality_measure == "WRAcc":
            quality = (tp+fp)/self.N*(tp/(tp+fp) - self.TP/self.N)
        elif self.quality_measure == "Qc":
            quality = tp-self._additional_params_for_quality_measure["c"]*fp
        elif self.quality_measure == "Qg":
            quality = tp / (fp + self._additional_params_for_quality_measure["g"])
        if report_tp_fp:
            return quality, tp, fp
        return quality
    
    def _column_exclusive_product(self,A):
        # Compute prefix cumulative product and shift it down by one row.
        prefix = torch.cumprod(A, dim=0)
        prefix = torch.roll(prefix, shifts=1, dims=0)
        prefix[0, :] = 1  # No elements above the first row
        # Compute suffix cumulative product by first flipping A,
        # computing the cumprod, then rolling and flipping back.
        suffix = torch.cumprod(A.flip(0), dim=0)
        suffix = torch.roll(suffix, shifts=1, dims=0).flip(0)
        suffix[-1, :] = 1  # No elements below the last row
        # Multiply prefix and suffix to get the product excluding the current element.
        return prefix * suffix
    
    def backward(self, lr=0.01):
        # Recompute g and W1
        self.forward()
        if self.quality_measure == "PPV":
            # Precompute the sum of g along its first dimension (over rows), for each u.
            sum_g = torch.sum(self.g, dim=0)  # shape: (k,)
            T0 = self.target[0]  # shape: (rows,)
            # Compute the weighted sum: for each u, sum over rows of T0 * g[:, u].
            sum_Tg = torch.sum(self.g * T0.unsqueeze(1), dim=0)  # shape: (k,)
            # Vectorized computation of dL_dg:
            # dL_dg[u, s] = -( T0[s]*sum_g[u] - sum_Tg[u] ) / (sum_g[u]**2)
            dL_dg = -((sum_g[:, None] * T0[None, :]) - sum_Tg[:, None]) / (sum_g[:, None]**2)
            dL_dg = dL_dg.to(torch.double)
        elif self.quality_measure == "WRAcc":
            dL_dg_single = -1/self.N * (self.target[0, :] - self.TP/self.N)
            dL_dg = dL_dg_single.unsqueeze(0).repeat(self._k, 1)
            dL_dg = dL_dg.to(torch.double)
        elif self.quality_measure == "Qc":
            dL_dg = self._additional_params_for_quality_measure["c"]*(1-self.target) - self.target
            dL_dg = dL_dg.repeat(self._k, 1)
            dL_dg = dL_dg.to(torch.double)
        elif self.quality_measure == "Qg":
            T0 = self.target[0]
            # Precompute weighted sums
            sum_g = torch.sum(self.g, dim=0).unsqueeze(1)
            sum_Tg = torch.sum(self.g * T0.unsqueeze(1), dim=0).unsqueeze(1)
            sum_1_Tg = torch.sum(self.g * (1 - T0).unsqueeze(1), dim=0).unsqueeze(1)
            dL_dg = (sum_Tg - self.target*(sum_g+self._additional_params_for_quality_measure["g"])) / ((sum_1_Tg + self._additional_params_for_quality_measure["g"])**2)
            dL_dg = dL_dg.to(torch.double)
        W1 = self.W1.unsqueeze(2).repeat(1, 1, self.N)
        x1 = self._selector_matrix.matrix.unsqueeze(1).repeat(1, self._k, 1)
        prod_W1 = (1-W1*(1-x1))
        dg_dW1 = self._column_exclusive_product(prod_W1)
        dg_dW1 = (self._selector_matrix.matrix-1).unsqueeze(1).repeat(1,self._k,1)*dg_dW1
        dg_dW1 = dg_dW1.swapaxes(0,1)
        dW1_dW = self.W1 * (1 - self.W1)
        dW1_dW = dW1_dW.to(torch.double)
        dg_dW1 = dg_dW1.to(torch.double)
        dL_dW = torch.einsum('um,usm,su->us', dL_dg, dg_dW1, dW1_dW)
        # If the gradient is zero, return
        if torch.all(abs(dL_dW) < 1e-100):
            # print("dL_dW is zero")
            return dL_dW
        # Normalize the gradient
        dL_dW = dL_dW / torch.norm(dL_dW, p=2)
        dL_dW = dL_dW.T
        # Update weights
        self.W = self.W - lr * dL_dW
        return dL_dW
    
    def get_subgroup_descriptions(self, threshold=0.5):
        if self.W1 is None:
            raise ValueError("Selector relevancy weights have not been computed yet. Please fit the model first.")
        if self._selector_matrix is None:
            raise ValueError("Selector matrix has not been computed yet. Please fit the model first.")
        subgroups = (self.W1 > threshold).to(torch.int).T
        return [
            self._selector_matrix.get_subgroup_description(subgroups[i,:]) for i in range(subgroups.shape[0])
        ]