from typing import List, Tuple

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

    def __init__(self, n_outputs, n_classes, y, sample_weight, weighted_n_samples, samples, start, end):
        """Initialize the criterion at node samples [start:end] and children
        samples[start:start] and samples[start:end].
        """
        self.sample_weight = None

        self.samples = None
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sum_total = None
        self.sum_left = None
        self.sum_right = None
        self.n_classes = n_classes

        self.sum_stride = n_classes

        #:#sum_stride = 0

        #:#for k in range(n_outputs):
        #:#    self.n_classes[k] = n_classes[k]

        #:#    if n_classes[k] > sum_stride:
        #:#        sum_stride = n_classes[k]

        #:#self.sum_stride = sum_stride

        # init
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        # read samples
        w = 1.0

        for p in range(start, end):
            s = samples[p]

            if sample_weight:
                w = sample_weight[i]

            c = self.y[s]
            #:#for k in range(self.n_outputs):
            #:#    c = self.y[i, k]
            #:#    sum_total[k * self.sum_stride + c] += w


class Entropy(ClassificationCriterion):
    """Entropy index impurity criterion.
    """
    def node_impurity(self):
        #NOTE n_classes currently int, can be upgraded to list for several outputs
        n_classes = self.n_classes
        sum_total = self.sum_total
        entropy = 0.0
        count_k = 0.0

        for c in range(n_classes):
            class_count = sum_total[c]
            if class_count > 0.0:
                #:# class_count /= self.weighted_n_node_samples
                entropy -= class_count * log(class_count)

            sum_total += self.sum_stride

        return entropy / self.n_outputs

    def children_impurity(self):

