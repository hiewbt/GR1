import numpy as np
from scipy.special import comb

def fuzzy_partition_coefficient(U):
    N = U.shape[1]
    F_c = np.sum(U**2) / N
    return F_c

def partition_entropy(U):
    N = U.shape[1]
    U_log_U = np.where(U > 0, U * np.log(U), 0)
    H_c = - np.sum(U_log_U) / N
    return H_c

def calinski_harabasz_index(X, U):
    labels = np.argmax(U, axis=0)

    n_clusters = len(np.unique(labels))
    n_samples = X.shape[0]
    overall_mean = np.mean(X, axis=0)

    between_dispersion = 0
    for k in np.unique(labels):
        cluster_k = X[labels == k]
        cluster_mean = np.mean(cluster_k, axis=0)
        between_dispersion += len(cluster_k) * np.sum((cluster_mean - overall_mean) ** 2)

    within_dispersion = 0
    for k in np.unique(labels):
        cluster_k = X[labels == k]
        cluster_mean = np.mean(cluster_k, axis=0)
        within_dispersion += np.sum((cluster_k - cluster_mean) ** 2)

    if within_dispersion == 0:
        return 0.0
    score = (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))
    return score



# a = tp    same class, same cluster
# d = tn    different class, different cluser
# b = fn    same class, different cluster
# c = fp    different class, same cluster

def compute_confusion(true_labels, U):
    pred_labels = np.argmax(U, axis=0)
    clusters = pred_labels
    classes = true_labels
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    M = comb(len(A), 2)
    tn = M - tp - fp - fn
    return tp, fp, fn, tn, M

def rand_index(true_labels, U):
    tp, fp, fn, tn, M = compute_confusion(true_labels, U)
    return (tp + tn) / (tp + fp + fn + tn)

def adjusted_rand_index(true_labels, U):
    tp, fp, fn, tn, M = compute_confusion(true_labels, U)
    numerator = tp - (tp + fp)*(tp + fn)/M
    denominator = (tp + fp + tp + fn)/2 - (tp + fp)*(tp + fn)/M
    return numerator/denominator

def jaccard_coefficient(true_labels, U):
    tp, fp, fn, tn, M = compute_confusion(true_labels, U)
    return tp / (tp + fn + fp)
