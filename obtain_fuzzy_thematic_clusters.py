import numpy as np
from lapin import lapin
from faddis import faddis

from operator import itemgetter

NUM_EL = 15
alpha = 0.2


if __name__ == "__main__":
    relevance_matrix = np.loadtxt("relevance_matrix.txt").T
    print(relevance_matrix.shape)

    rm_threshold = relevance_matrix > alpha
    abstract_threshold_count = np.sum(rm_threshold, axis=0)
    max_threshold_count = max(abstract_threshold_count)
    abstract_weight = max_threshold_count / abstract_threshold_count
    W = np.diag(abstract_weight)

    tc = relevance_matrix.dot(W).dot(relevance_matrix.T)
    print(tc.shape)

    tc_transformed = lapin(tc)
    B, member, contrib, intensity, lat, tt = faddis(tc_transformed)
    np.savetxt("clusters.dat", member)

    with open("taxonomy_leaves.txt") as fn:
        annotations = [l.strip() for l in fn]

    for cluster in member.T:
        print(list(sorted(zip(annotations, cluster.flat),
                          key=itemgetter(1), reverse=True))[:NUM_EL])

