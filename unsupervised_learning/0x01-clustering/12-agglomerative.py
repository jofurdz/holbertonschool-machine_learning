#!/usr/bin/env python3
"""Contains the function agglomerative()"""

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset"""
    sch = scipy.cluster.hierarchy

    link = sch.linkage(X, method='ward')
    clss = sch.fcluster(Z=link, t=dist, criterion='distance')
    dend = sch.dendrogram(link, color_threshold=dist)

    plt.show()

    return clss
