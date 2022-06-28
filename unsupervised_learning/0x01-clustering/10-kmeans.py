#!/usr/bin/env python3
"""Contains the function kmeans()"""

import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
