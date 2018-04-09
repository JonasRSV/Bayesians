from numpy import array, zeros, mean, log, ones, square, diag
from sys import exit, stdout

MINUS_INF = -100000000000
PLUS_INF = 100000000000

class memory(object):
    """Store trained data."""

    def __init__(self, ev=None, cov=None, p=None, ssz=None, cl=None):
        """
        Store values.

        Expected Values: ev
        Covariances: cov
        Prioris: p
        Sample Size: ssz
        classes: cl
        """
        self.ev = ev
        self.cov = cov
        self.p = p
        self.cl = cl


def ml_norm_dist(train, labels, bw=None):
    """Calculate Maximum likelihood.

    These ML variables are calculated with respect
    to the log ML of the generalized normal distribution.
    """
    if bw is None:
        bw = ones(len(labels))

    classes = set(labels)
    class_sz = len(classes)
    _, features_sz = train.shape

    ev_shape = (class_sz, features_sz)
    cov_shape = (features_sz, features_sz)
    covs_shape = (class_sz, features_sz, features_sz)

    expected_values = zeros(ev_shape)
    covariances = zeros(covs_shape)

    for class_ in classes:
        class_ = int(class_)

        class_data = train[class_ == labels]
        weights = bw[class_ == labels]

        expected_values[class_] = (weights.T @ class_data) / sum(weights)
        covariance = zeros(cov_shape)

        for feature in range(features_sz):
            variance = square(class_data[:, feature]
                              - expected_values[class_][feature])

            covariance[feature][feature] =\
                (weights @ variance) / sum(weights)

        covariances[class_] = covariance

    return covariances, expected_values


def norm_dist_predict(mem, X):
    """Predict belongance of X.

    using the generalized normal distribution.
    """
    class_belongance = None

    """Really low number."""
    max_aposteori = -100000000000

    for class_ in mem.cl:
        class_ = int(class_)

        cov = mem.cov[class_]
        ev = mem.ev[class_]
        priori = mem.p[class_]

        quadratic_form = (X - ev).T @ inv(cov) @ (X - ev)

        """0 Variance features will give log 0, which is error."""
        log_likelyhood = -0.5 * (log(det(cov) + 1e-10) + quadratic_form)
        log_priori = log(priori)

        belongance = log_likelyhood + log_priori

        if belongance > max_aposteori:
            max_aposteori = belongance
            class_belongance = class_

    return class_belongance


def det(matrix):
    """Get determinant of a diagonalized matrix (Or assumed to be)."""
    det = 1
    for i in diag(matrix):
        det *= i

    return det


def inv(matrix):
    """Invert a diagonalized matrix."""
    inverted = zeros(matrix.shape)
    for i, var in enumerate(diag(matrix)):
        """
        If variance is 0 there's something wierd
        with the training data, might not even
        need a classifier for this. But ill
        add a check here so that the program
        does not crash."""

        if var == 0:
            inverted[i][i] = 10000000
        else:
            inverted[i][i] = 1 / var

    return inverted


class classifier(object):
    """Classifier class."""

    def __init__(self, m=None, ml=ml_norm_dist, p=norm_dist_predict):
        """Constructor.

        Memory: m
        Maximum Likelyhood: ml
        predictor: p
        """
        self.m = m
        self.ml = ml
        self.p = p

    def __calculate_priori(self, labels, bw=None):
        """Calculate the priori for each class."""
        classes = set(labels)
        class_sz = len(classes)
        samples = len(labels)

        if bw is None:
            bw = ones(samples)

        priors = zeros(class_sz)

        """Label ones."""
        lo = ones(samples)

        for class_ in classes:
            class_ = int(class_)

            occurrences = lo[class_ == labels]
            weights = bw[class_ == labels]

            priors[class_] = (weights @ occurrences) / sum(bw)

        return priors

    def train(self, data, labels, bw=None):
        """Train the classifier."""
        self.m = memory()
        self.m.cl = set(labels)
        self.m.p = self.__calculate_priori(labels, bw)
        self.m.cov, self.m.ev = self.ml(data, labels, bw)

        return self

    def predict(self, X):
        """Predict new data."""
        return self.p(self.m, X)

