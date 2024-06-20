import numpy as np
from sklearn.metrics import pairwise_distances


def fuzzy_partition_coefficient(U):
    C, N = U.shape
    numerator = np.sum(U ** 2)
    denominator = C * N
    return numerator / denominator

def partition_entropy(U):
    pe = U * np.log2(U)
    pe = np.sum(pe, axis=0)
    pe = -np.mean(pe, axis=-1)
    return pe

def calinski_harabasz_index(X, U, V):
    U=U.T
    n_samples, n_features = X.shape
    n_clusters = V.shape[0]

    # Tính trung bình toàn bộ của dữ liệu
    overall_mean = np.mean(X, axis=0)

    # Tính tổng bình phương giữa các cụm (SSB)
    SSB = 0.0
    for j in range(n_clusters):
        n_cluster_points = np.sum(U[:, j])
        cluster_mean = V[j]
        SSB += n_cluster_points * np.sum((cluster_mean - overall_mean) ** 2)

    # Tính tổng bình phương trong mỗi cụm (SSW)
    SSW = 0.0
    for j in range(n_clusters):
        cluster_points = X - V[j]
        SSW += np.sum(U[:, j] * np.sum(cluster_points ** 2, axis=1))

    # Tính chỉ số Calinski-Harabasz (VRC)
    VRC = (SSB / (n_clusters - 1)) / (SSW / (n_samples - n_clusters))
    return VRC

def davies_bouldin_index(X, U, V):
    U = U.T
    n_samples, n_features = X.shape
    n_clusters = V.shape[0]

    # Tính sự phân tán (scatter) của từng cụm
    S = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = X - V[i]
        S[i] = np.sqrt(np.sum(U[:, i] * np.sum(cluster_points ** 2, axis=1)) / np.sum(U[:, i]))

    # Tính khoảng cách giữa các cụm
    M = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            M[i, j] = M[j, i] = np.linalg.norm(V[i] - V[j])

    # Tính chỉ số Davies–Bouldin cho mỗi cụm
    R = np.zeros(n_clusters)
    for i in range(n_clusters):
        R[i] = np.max([(S[i] + S[j]) / M[i, j] for j in range(n_clusters) if i != j])

    # Tính Davies–Bouldin Index
    DBI = np.mean(R)
    return DBI