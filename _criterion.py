from typing import List, Tuple

class Criterion:
    """Interface for impurity criteria.

       This object stores methods on how to calculate how good a split is using
       different metrics.
    """
    def __init__(self,
                 y: List,
                 sample_weight: List[float],
                 weighted_n_samples: float,
                 samples,
                 start,
                 end):
        pass

    def reset(self):
        """Placeholder

        Reset criterion at pos=start.
        """
        pass

    def reverse_reset(self):
        """Placeholder

        Reset criterion at pos=end.
        """
        pass

    def update(self, new_pos: int):
        """Placeholder

        Update statistics by moving samples[pos:new_pos] from the right
        child to the left child.
        """
        pass

    def node_impurity(self):
        """Placeholder

        Calculates node impurity, i.e. calculate impurity from [start:end]
        """
        pass

    def children_impurity(self) -> Tuple[float, float]:
        """Placeholder

        Calculates children impurity, i.e.[start:pos] and [pos:end]
        """
        pass

    def node_value(self) -> float:
        """Placeholder

        Calculates node value, i.e.[start:pos] and [pos:end]
        """
        pass

    #NOTE: needed for pruning
    def proxy_impurity_improvement(self) -> float:
        pass

    #NOTE: needed for pruning
    def impurity_improvement(self):
        pass

class ClassificationCriterion(Criterion):
    """Abstract class for classification Criterion."""

    def __init__(self, n_outputs, n_classes, y, sample_weight, weighted_n_samples, samples, start, end):
        """Initialize the criterion at node samples [start:end] and children
        samples[start:start] and samples[start:end].
        """
        #NOTE: n_outputs will be 1 until multiple outputs are supported
        # cinit
        n_outputs = 1

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
