import numpy as np
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

    np.savetxt("corelevance_matrix.txt", tc)
