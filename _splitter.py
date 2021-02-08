# Python imports
import numpy as np
from collections import namedtuple

# Forst imports
from _utils import rand_int

INFINITY = float('inf')
MINFINITY = float('-inf')

class SplitRecord():
    def __init__(self, pos):
        self.impurity_left = INFINITY
        self.impurity_right = INFINITY
        self.pos = pos
        self.feature = 0
        self.threshold = 0.
        self.improvement = MINFINITY

class Splitter():
    """Abstract Splitter class, used by both Base and Sparse Splitter.

    Parameters:
    criterion : Criterion
        Criterion function to measure quality of split.
    max_features : int
        Max number of randomly selected features which can be considered for a
        split.
    min_samples_leaf : int
        Min number of samples that each leaf can have.
    min_weight_leaf : float
        Min leaf weight each leaf can have. The leaf weight is the sum of the
        weights of each sample in the leaf.
    X : np.ndarray
        This object contains the inputs. 2d array.
    y : np.ndarray
        Vector of targets or true labels, per samples.
    sample_weight : np.ndarray
        Weights of the samples. If not provided, uniform weight is assumed.
    """
    def __init__(self, criterion, max_features, min_samples_leaf,
                 min_weight_leaf, X, y, sample_weight=None):
        # samples are rows of X
        self.samples = []

        n_samples = X.shape[0]
        weighted_n_samples = 0.0
        for i in range(n_samples):
            # only work with positively weighted samples
            if sample_weight == None or sample_weight[i] >= 0.0:
                self.samples.append(X[i])

            # weighted_n_samples have a total weighted sum of samples
            if sample_weight != None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # n_samples could have less than the actually used samples
        self.n_samples = len(self.samples)
        self.weighted_n_samples = weighted_n_samples
        self.sample_weight = sample_weight

        # features are cols of X
        self.n_features = X.shape[1]
        self.features = list(range(self.n_features))

        self.feature_values = [None for _ in range(self.n_samples)]
        self.constant_features = [None for _ in range(self.n_features)]
        self.y = y

    def node_reset(self, start, end, weighted_n_node_samples):
        """Reset splitter on node samples[st:en].
        """
        self.start = start
        self.end = end
        self.criterion.init(self.y,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        return self.criterion.weighted_n_node_samples

    def node_value(self):
        return self.criterion.node_value()

    def node_impurity(self):
        return self.criterion.node_impurity()


class BaseDenseSplitter(Splitter):
    def __init__(self, criterion, max_features, min_samples_leaf,
                 min_weight_leaf, X, y, sample_weight=None):
        super.__init__(criterion, max_features, min_xamples_leaf,
                       min_weight_leaf, X, y, sample_weight)
        self.X = X


class BestSplitter(BaseDenseSplitter):
    """Find the best split on node samples[start:end]."""
    def node_split(impurity: float, sr: SplitRecord, n_constant_features):
        samples = self.samples
        start = self.start
        end = self.end

        features = self.features
        constant_features = self.constant_features
        n_features = self.n_features

        Xf = self.feature_values
        max_features = self.max_features
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf
        # random state

        # best and current SplitRecord
        current_proxy_improvement = MINFINITY
        best_proxy_improvement = MINFINITY

        p = 0
        feature_idx_offset = 0
        feature_offset = 0
        i = 0
        j = 0

        n_visited_features = 0
        current_feature_value
        partition_end = 0

        # best Splitrecord yet
        best = SplitRecord(end)

        # Number of features discovered to be constant during the split search
        n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        n_drawn_constants = 0
        n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        n_total_constants = n_known_constants
        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        # f_i == features to try
        f_i = n_features
        f_j = 0
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        # Continue if:
        #   1) all remaining features are constant
        #      2) we've visited our max allowed features to search
        #      3) at least one drawn feature must be a constant
        while (f_i > n_total_constants and \
                (n_visited_features < max_features or \
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            n_visited_features += 1
            # draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants:n_known_constants[
                # swap feature[f_j] with feature[n_total_constants]
                # increase n_drawn_constants by 1, updating n_total_constants?
                features[f_j], features[n_total_constants] = \
                                    features[n_total_constants], features[f_j]
                n_drawn_constants += 1

            else:
                # f_j in the interval
                # [n_known_constants:f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants:f_i[



