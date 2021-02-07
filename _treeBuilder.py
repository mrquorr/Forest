# Python imports
import numpy as np
from typing import Optional

# Forest imports
from _splitter import SplitRecord

class TreeBuilder():
    def build(self, tree: Tree, X: np.ndarray, y: np.ndarray,
              sample_weight: Optional[np.ndarray] = None):
        """Place holder for build function used by DepthFirstTreeBuilder and
        BestFirstTreeBuilder

        Parameters:
        tree : Tree
            initialized tree to build
        X : np.ndarray
            X dataset
        y : np.ndarray
            y dataset
        sample_weight : Optional[np.ndarray]
            Array with sample weight for each sample.
            Uniform sample by default (None).
        """
        pass

class DepthFirstTreeBuilder(TreeBuilder):
    # LEAF rules: min_samples_split, min_samples_leaf, min_weight_leaf
    def __init__(self, splitter: Splitter,
                 min_samples_split: Optional[int] = None,
                 min_samples_leaf: Optional[int] = None,
                 min_weight_leaf: Optional[float] = None,
                 max_depth: Optional[int] = None,
                 min_impurity_decrease: Optional[float] = None,
                 min_impurity_split: Optional[float] = None):
        """ Build Trees using the defined configuration

        Parameters:
        splitter : Splitter
            Algorithm that decides which split of the considered splits is
            best {best, random}.

        LEAF RULES (any split violating this restrictions will be skipped):
        min_samples_leaf : int
            mininimum accepted samples per leaf
        min_weight_leaf : float
            minimum accepted weight per leaf
        max_depth : int
            maximum depth allowed
        min_impurity_decrease : float
            minimum accepted impurity decrease
        min_impurity_split : float
            minimum accepted impurity
        """
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    def build(self, tree: Tree, X: np.ndarray, y: np.ndarray,
              sample_weight: Optional[np.ndarray] = None):
        """Build a tree using X, y and depth first rules for node-splitting.

        Parameters:
        see TreeBuilder
        """
        splitter = self.splitter
        min_samples_split = self.min_samples_split
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf
        max_depth = self.max_depth
        min_impurity_decrease = self.min_impurity_decrease
        min_impurity_split = self.min_impurity_split

        # init()
        # n_samples
        # node_impurity()
        # node_split()
        splitter.init(X, y, sample_weight_ptr)
        max_depth_seen = -1
        # with nogil:
        # each StackRecord has:
        # start, end, depth, parent, is_left, impurity, n_constant_features
        n_node_samples = splitter.n_samples
        stack = [0, n_node_samples, 0, None, None, INFINITY, 0)]

        # building variables
        is_leaf = False
        first = True
        max_depth_seen = -1

        while stack:
            start, end, depth, parent, is_left, impurity, \
                                            n_constant_features = stack.pop()

            # number of entries in mask minus the masked values
            n_node_samples = end - start
            # TODO: weighted_n_node_samples = splitter.node_reset()?
            splitter.node_reset(start, end, weighted_n_node_samples)

            # node is leaf when...
            # 1) depth is greater than max_depth
            # 2) samples in the node are less than min_samples_split
            # 3) samples in the node are less than 2 * min_samples_leaf
            # 4) weighted samples are less than 2 * min_weight_leaf
            is_leaf = (depth >= max_depth or
                       n_node_samples < min_samples_split or
                       n_node_samples < 2 * min_samples_leaf or
                       weighted_n_node_samples < 2 * min_weight_leaf)

            # NOTE: added for case where the initial dataset is already pure
            #       enough
            if first:
                impurity = splitter.node_impurity()
                first = False

            # node can also be a leaf when...
            # 5) impurity is less than or equal to min_impurity_split
            is_leaf = is_leaf or impurity <= min_impurity_split

            if not is_leaf:
                # sr = _splitter.py: SplitRecord
                sr = splitter.node_split(impurity, n_constant_features)
                # after splitting, we can determine the node should actually be
                # a leaf if...
                # 6) the split position is at the end
                # 7) improvement is below the min_impurity_decrease
                is_leaf = is_leaf or sr.pos >= end or \
                          sr.improvement < min_impurity_decrease

            node_id = tree._add_node(parent, is_left, is_leaf, sr.feature,
                                     sr.threshold, impurity, n_node_samples,
                                     weighted_n_node_samples)


            # Store value for all nodes, to facilitate tree/model
            # inspection and interpretation.
            # TODO figure out, function copies splitter node value to tree
            #:# splitter.node_value(tree.value + node_id * tree.value_stride)

            if not is_leaf:
                # push right child
                stack.push(tuple(sr.pos, end, depth + 1, node_id, False,
                                 sr.impurity_right, n_constant_features))
                # push left child
                stack.push(tuple(start, sr.pos, depth + 1, node_id, True,
                                 sr.impurity_left, n_constant_features))

            if depth > max_depth_seen:
                max_depth_seen = depth
        tree.max_depth = max_depth_seen
