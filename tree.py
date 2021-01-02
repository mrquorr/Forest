#!/home/mrquorr/anaconda3/bin/python3

from ._criterion import Criterion

clf = 'clf'
reg = 'reg'

CRITERIA_CLF = {"gini": _criterion.Gini,
                "entropy": _criterion.Entropy}

CRITERIA_REG = {"mse": _criterion.MSE,
                "friedman_mse": _criterion.FriedmanMSE,
                "mae": _criterion.MAE,
                "poisson": _criterion.Poisson}

CRITERIA = {clf, CRITERIA_CLF,
            reg, CRITERIA_REG}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter,
                   "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {"best": _splitter.BestSparseSplitter,
                    "random": _splitter.RandomSparseSplitter}

class DesicionTree():
    def __init__(self,
                 criterion="gini",
                 splitter="depth",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_inpurity_decrease=0.,
                 class_weight=None,
                 ccp_alpha=0.0,
                 is_classifier=False):
        self.criterion = criterion
        self.splitter = splitters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_inpurity_decrease = min_inpurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

        if is_classifier:
            self.pred_type = clf
        else:
            self.pred_type = reg
        # -- after fit --
        # self.n_features_
        # self.n_features_in_
        # self.tree_
        # self.classes_
        # self.n_classes_

    def is_classifier(self):
        return self.pred_type == clf

    def get_depth(self):
        # check if model is fitted, otherwise tree is not created
        return self.tree_.max_depth

    def get_n_leaves(self):
        # check if model is fitted, otherwise tree is not created
        return self.tree_.n_leaves

    def fit(self,
            X,
            y,
            sample_weight=None,
            check_input=True):
        #:# random_state = check_random_state(self.random_state)

        #:# if self.ccp_alpha < 0.0:
        #:#     raise ValueError("ccp_alpha must be greater than or equal to 0")

        # check input
        if check_input:
            # input shape
            # y := [[n_samples, ]]
            # X := [[n_samples, n_features]]
            # validate X, y shapes

        # output shape
        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_

        #:# y = np.atleast_1d(y)
        #:# expanded_class_weight = None

        #:# if y.ndim == 1:
        #:#     # reshape is necessary to preserve the data contiguity against vs
        #:#     # [:, np.newaxis] that does not.
        #:#     y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        # set classifier only args
        if self.is_classifier:
            self.classes_ = []
            self.n_classes_ = []

            #TODO: add class_weight
            #:#if self.class_weight is not None:
            #:#    y_original = np.copy(y)

            # encode all possible y's to unique classes
            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
                y = y_encoded

            #TODO: add class weight
            #:# if self.class_weight is not None:
            #:#     expanded_class_weight = compute_sample_weight(self.class_weight, y_original)

        # check parameters
        max_depth = float('inf') if not self.max_depth else self.max_depth
        max_leaf_nodes = -1 if not self.max_leaf_nodes else self.max_leaf_nodes

        # min_samples_leaf
        # min_samples_split
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        # max_features
        self.max_features_ = max_features

        args = (self.n_outputs_, self.n_classes_)
        # TODO create CRITERION class
        criterion = CRITERIA[self.pred_type][self.criterion](self.n_outputs_, self.n_classes_)

        # TODO create SPLITTER class
        #:#SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS
        SPLITTERS = DENSE_SPLITTERS
        splitter = SPLITTERS[self.splitter](criterion,
                                            self.max_features_,
                                            min_samples_leaf,
                                            min_weight_leaf,
                                            random_state)

        # TODO create TREE class
        # NOTE: REGRESSION tree shouldn't need 2nd arg
        self._tree = Tree(self,n_features_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        # TODO create BUILDER class
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                            min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth,
                                            self.min_impurity_decrease,
                                            min_impurity_split)
        else:
            builder = BestFirstTreeBuilder(splitter, min_samples_split,
                                            min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth,
                                            self.min_impurity_decrease,
                                            min_impurity_split)

        builder.build(self.tree_, X, y, sample_weight)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()
        return self
