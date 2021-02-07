import numpy as np

# TODO tree._add_node
# TODO tree.node_value
# TODO tree._resize_c
class Node():
    """Node containing split information
    """
    def __init__(self):
        self.impurity = 0.0
        self.mask: List[bool] = []

class Tree():
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a list of Nodes. Each Node holds information
    on a split.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    def __init__(self, n_features: int, n_classes: int, n_outputs:int=1):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_outputs = n_outputs

        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = None
        self.nodes = []

    def _add_node(self, parent: int, is_left: bool, is_leaf:bool,
                  feature: int, threshold: float, impurity: float,
                  n_node_samples: int, weighted_n_node_samples: float):
        """Add a node to the tree.
        The new node registers itself as the child of its parent.
        """
        node_id = self.node_count
        #:# no need to check for allocation since this isn't cpython
        #:#if node_id >= self.capacity:
        #:#    # try to resize
        #:#    # return SIZE_MAX if failed



