from typing import List, Tuple
from libc.math

class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """
    def __init__(self, y, sample_weight, weighted_n_samples, samples, start,
                 end):
        """
        Parameters:
        y : List[float]
            Buffer to store values for n_outputs target variables.
        sample_weight : List[float]
            Weight of each sample.
        weighted_n_samples : float
            Total weight of samples being considered.
        samples : List[int]
            Indices of samples in X and y, where samples[start:end]
            correspond to the samples in this node.
        start
            First sample to be used in this node.
        end
            Final sample to be used in this node.
        """
        pass


    def reset(self):
        """Reset criterion at pos=start.
        Needs to be implemented by subclass.
        """
        pass

    def reverse_reset(self):
        """Reset criterion at pos=end.
        Needs to be implemented by subclass.
        """
        pass

    def update(self, new_pos: int):
        """Update statistics by moving samples[pos:new_pos] from the right
        child to the left child.
        This means that new_pos must be > than old_pos.
        Needs to be implemented by subclass.

        Parameters:
        new_pos : int
            New starting index position of the samples in the right child.
        """
        pass

    def node_impurity(self):
        """Calculates node impurity, i.e. calculate impurity from [start:end].
        This is the primary function of the Criterion class, the lower the
        impurity the better.
        Needs to be implemented by subclass.
        """
        pass

    def children_impurity(self) -> Tuple[float, float]:
        """Calculates children impurity, i.e.[start:pos] and [pos:end].
        NOTE: Original function modified args.
        Needs to be implemented by subclass.
        """
        pass

    def node_value(self) -> float:
        """Calculates node value for samples[start:end].
        NOTE: Original function modified args.
        Needs to be implemented by subclass.
        """
        pass

    def proxy_impurity_improvement(self) -> float:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        impurity_left, impurity_right = self.children_impurity()
        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)



    def impurity_improvement(self, impurity_parent, impurity_left,
                             impurity_right):
        """Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.

        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity_parent : float
            The initial impurity of the parent node before the split
        impurity_left : float
            The impurity of the left child
        impurity_right : float
            The impurity of the right child

        Return
        ------
        float : improvement in impurity after the split occurs
        """
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
            (impurity_parent - (self.weighted_n_right /
                                self.weighted_n_node_samples * impurity_right)
                             - (self.weighted_n_left /
                                self.weighted_n_node_samples * impurity_left)))

class ClassificationCriterion(Criterion):

    def __init__(self, n_outputs, n_classes, y, sample_weight,
                 weighted_n_samples, samples, start, end):
        """Initialize the criterion at node samples [start:end] and children
        samples[start:start] and samples[start:end].

        Parameters:
        n_outputs : int
            number of targets, dimentionality of prediction
        n_classes : np.ndarray
            unique number of classes in each target, has length of n_outputs
        y : np.ndarray
        sample_weight
        weighted_n_samples
        samples
        start
        end
        """
        self.samples = samples
        self.sample_weight = sample_weight
        self.start = start
        self.end = end
        self.pos = 0

        self.n_samples = 0
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.n_classes = n_classes
        self.n_outputs = n_outputs
        self.max_stride = max(n_classes)

        # max_stride: max cardinality (classes) for the different targets
        # n_outputs: number of targets
        self.n_elements = self.n_outputs * self.max_stride
        # sum lists hold for every class max_stride available spaces to store
        # information on observed classes

        # ex:
        # 4 targets with 3, 2, 1, 1 classes respectively
        # this example will have for each target 3 number counters to keep
        # track of weighted presence of class

        self.sum_total = [0.0 for _ in range(self.n_elements)]
        self.sum_left = [0.0 for _ in range(self.n_elements)]
        self.sum_right = [0.0 for _ in range(self.n_elements)]

        self.y = y

        # update sum_total with information for all n samples
        w = 1.0
        for p in range(start, end):
            i = samples[p]

            if sample_weight:
                w = sample_weight[i]

            # w will be the i'th sample's weight or 1.0 for the case of
            # uniform distribution
            for k in range(self.n_outputs):
                c = self.y[i, k]
                self.sum_total[k * self.max_stride + c] += w

            self.weighted_n_node_samples += w

        # set pos to start
        self.reset()

    def reset():
        """Reset position to start.
        After opertation:
            left values will have
                weighted_n : 0.0
                sum_left = [0.0 for each of n_elements]
            right values will have
                weighted_n : weighted_n_node_samples
                sum_right = list.copy(self.sum_total)
        """
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        # with reset, left child will have 0.0 for all target's classes and
        # the right child will have the count of all n samples
        self.sum_left = [0.0 for _ in range(self.n_elements)]
        self.sum_right = list.copy(self.sum_total)

    def reverse_reset():
        """Reset position to end.
        After opertation:
            right values will have
                weighted_n : 0.0
                sum_right = [0.0 for each of n_elements]
            left values will have
                weighted_n : weighted_n_node_samples
                sum_left = list.copy(self.sum_total)
        """
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        # with reset, right child will have 0 for all classes and left
        # child will have the same as the total
        self.sum_left = list.copy(self.sum_total)
        self.sum_right = [0.0 for _ in range(self.n_elements)]

    def update(new_pos):
        """Update stats by moving samples[pos:new_pos] to the left child.

        Parameters:
        """
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        w = 1.0
        # 1) pos -> new_pos
        if (new_pos - self.pos) <= (self.end - new_pos):
            for p in range(self.pos, new_pos):
                i = samples[p]
                if self.sample_weight:
                    w = self.sample_weight[i]
                for k in range(self.n_outputs):
                    # k: target index
                    # max_stride: max cardinality of all target classes
                    # y[i, k]: the kth target of the ith sample
                    label_ix = k * self.max_stride + self.y[i, k]
                    self.sum_left[label_ix] += w
                self.weighted_n_left += w
        # 2) end -> new_pos
        else:
            self.reverse_reset()

            for p in reversed(range(new_pos, end)):
                i = samples[p]
                if self.sample_weight:
                    w = sample_weight[i]
                for k in range(self.n_outputs):
                    label_ix = k * self.max_stride + self.y[i, k]
                    self.sum_left[label_ix] -= w
                self.weighted_n_left -= w

        # update right part by disjointing the total n sample stats from
        # the left stats for all targets and classes
        self.weighted_n_right = \
                            self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                label_ix = k * self.max_stride + c
                self.sum_right[label_ix] = \
                            self.sum_total[label_ix] - self.sum_left[label_ix]

        self.pos = new_pos

    def node_value(self):
        """Return the sum total value for the node for samples[start:end]"""
        return list.copy(self.sum_total)


def log(x):
    return math.log(x) / math.log(2.0)

class Entropy(ClassificationCriterion):
    r"""Cross Entropy impurity criterion.
    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let
        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)
    be the proportion of class k observations in node m.
    The cross-entropy is then defined as
        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """
    def node_impurity(self):
        """Evaluate the impurity of the current node.
        Evaluate the cross-entropy criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        entropy = 0.0
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                label_ix = k * self.max_stride + c
                count_k = sum_total[label_ix]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)

        return entropy / self.n_outputs

    def children_impurity(self):
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        entropy_left = 0.0
        entropy_right = 0.0
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                label_ix = k * self.max_stride + c
                # left aggregate
                count_k = self.sum_left[label_ix]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)
                # right aggregate
                count_k = self.sum_right[label_ix]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)

        entropy_left /= self.n_outputs
        entropy_right /= self.n_outputs

        return entropy_left, entropy_right


