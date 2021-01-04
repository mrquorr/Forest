import numpy as np

INFINITY = np.inf
# TODO splitter.node_reset
# TODO splitter.node_impurity
# TODO splitter.node_split
# TODO tree._add_node
# TODO tree.node_value
# TODO tree._resize_c

class TreeBuilder():
    """Interface for different tree building strategies."""

    def build(self, tree: 'Tree', X: np.ndarray, y: np.ndarray,
              sample_weight: np.ndarray=None):
        """Placeholder

        Build a decision tree from the training set (X, y)."""
        pass

    def _check_input(self, X: np.ndarray, y: np.ndarray,
                     sample_weight: np.ndarray=None):
        """Check input dtypa, layout and format"""
        # TODO complete function
        # raise error when check fails
        pass

class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __init__(self, splitter: 'Splitter',
                 min_samples_split: int,
                 min_samples_leaf: int,
                 min_weight_leaf: float,
                 max_depth: int,
                 min_impurity_decrease: float,
                 min_impurity_split: float):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    def build(self, tree: 'Tree',
              X: np.ndarray,
              y: np.ndarray,
              sample_weight: np.ndarray=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        self._check_input(X, y, sample_weight)

        # tree capacity
        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        # TODO do we need resize?
        #:# tree._resize(init_capacity)

        splitter = self.splitter
        max_depth = self.max_depth
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf
        min_samples_split = self.min_samples_split
        min_impurity_decrease = self.min_impurity_decrease
        min_impurity_split = self.min_impurity_split

        max_depth_seen = -1

        splitter.init(X, y, sample_weight_ptr)

        # TODO how is weighted_n_node_samples used?
        weighted_n_node_samples = 0.0

        first = 1

        # each StackRecord has:
        # start, end, depth, parent, is_left, impurity, n_constant_features
        n_node_samples = splitter.n_samples
        stack = [tuple(0, n_node_samples, 0, None, None, INFINITY, 0)]

        while stack:
            start, end, depth, parent, is_left, impurity, n_constant_features = stack.pop()

            n_node_samples = end - start
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

            if first:
                impurity = splitter.node_impurity()
                first = 0

            # node can also be a leaf when...
            # 5) impurity is less than or equal to min_impurity_split
            is_leaf = is_leaf or impurity <= min_impurity_split

            if not is_leaf:
                splitter.node_split

            # TODO split.feature, split.threshold, split var does not exist(SplitRecord)
            node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                     split.threshold, impurity, n_node_samples,
                                     weighted_n_node_samples)

            splitter.node_value(tree.value + node_id * tree.value_stride)

            if not is_leaf:
                stack.push(tuple(split.pos, end, depth + 1, node_id, False,
                                 split.impurity_right, n_constant_features))
                stack.push(tuple(left, split.pos, depth + 1, node_id, True,
                                 split.impurity_right, n_constant_features))

            if depth > max_depth_seen:
                max_depth_seen = depth
        tree._resize_c(tree.node_count)
        tree.max_depth = max_depth_seen
