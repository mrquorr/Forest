FOREST
======
.
Tree (classification and regression) and Forest implementation in Python,
simplified from the Sci-Kit learn classes for learning and explaining purposes.

References
==========
* [1 wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
* [2 SK Learn](https://github.com/scikit-learn/scikit-learn/tree/42aff4e2edd8e8887478f6ff1628f27de97be6a3/sklearn/tree)
* [3 T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical Learning", Springer, 2009.](https://www.amazon.com/The-Elements-of-Statistical-Learning/dp/0387848843)

TreeBuilder
===========
Build any number of trees given an initial configuration.
- TreeBuilder

   - DepthFirstTreeBuilder
     + init()
       * has rules for defining leaf nodes defined by LEAF rules
       * has rules for when to stop splitting the tree defined by HALT rules
     + build()
       * has a stack with information for the first node
       * The first entry in the stack is manually added, it contains
         information for the first node. If the first node is not a leaf node,
         it'll get the best split and push each child to the stack with the
         impurity information obtained during the parent's turn.
       * while the stack is not empty:
          1: get from the stack information for considered node
          2: split the node with the previously calculated impurity
          3: push to stack both left and right children with their
             corresponding impurity

   - BestFirstTreeBuilder

Splitter
========
Keep track of splits with:
- SplitRecord
  + feature        # which feature to split on
  + pos            # split samples at pos,
                   # i.e. count of samples below the threshold for feature
                   # pos is >= end if node is a leaf
  + threshold      # threshold to split at
  + improvement    # impurity improvement given parent node
  + impurity-left  # impurity of the left split
  + impurity-right # impurity of the right split

Decides which node is the best to split based on CRITERION impurity.
- Splitter
   + init()
   + reset-splitter()

   - BaseDenseSplitter
      + init()

      - BestSplitter
        + node_split(impurity, SplitRecord, n_constant_features)
          * sample features randomly skipping known_constant_features
          * store calculations for efficiency
      - RandomSplitter

   - BaseSparseSplitter
      - BestSparseSplitter
      - RandomSparseSplitter

Criterion
=========
Function to calculate impurity for each possible node.
Criterion
  - ClassificationCriterion
    - Entropy
    - Gini
  - RegressionCriterion
    - MSE
    - MAE
    - FriedmanMSE
    - Poisson

Tree
====
Binary tree structure where each node has conditions dividing the samples into
2 groups. At the end of the rule chain there's a label identifying the sample's
final label (y).
This means that given a a Tree trained with a given dataset (X, y) will have
at each leaf node a prediction for y based on observations from (X, y).
1. from initial dataset (could be a subset of available data), split node
2. until HALT, continue splitting nodes based on SPLITTER, CRITERION, and LEAF
-HALT rules
-LEAF rules {max-depth, min-samples-split, min-samples-leaf,
             min_weight_leaf, min_impurity_split, min_impurity_decrease}
+ dataset mask (initial dataset copy)
+ collection of NODES
Tree:
  - ClassificationTree
  - RegressionTree

Node
====
Base storage for Node information.
+ left-child: index
+ right-child: index
+ feature: index
+ threshold: float
+ impurity: float
+ n-node-samples: int? 
+ weighted-n-node-samples: float? 

+ dataset mask
+ dataset?
-FEATURE {categorical, numeric}
-THRESHOLD {categorical, numeric}
-LABEL(S) if LEAF

BUILDING A FOREST
=================
* have N trees each trained on a random subset of training samples AND
  random subset of features

GETTING FOREST READY FOR PRODUCTION
===================================
* serialize forest into quick read file
* module to serialize input stream to feature sample
* be able of predicting input with serialized FOREST
* (skip tree grower part for production)
