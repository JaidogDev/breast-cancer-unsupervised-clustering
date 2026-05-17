import numpy as np

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)


def cluster_purity(y_true, y_pred):
    """
    Compute cluster purity using true labels only after clustering.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def safe_silhouette_score(X, labels):
    """
    Silhouette requires at least 2 clusters.
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    return silhouette_score(X, labels)


def evaluate_clustering(X, y_true, y_pred):
    """
    Evaluate clustering result.

    Silhouette Score does not require true labels.
    ARI, NMI, and Purity use true labels only after clustering.
    """
    return {
        "silhouette_score": float(safe_silhouette_score(X, y_pred)),
        "adjusted_rand_index": float(adjusted_rand_score(y_true, y_pred)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, y_pred)),
        "purity": float(cluster_purity(y_true, y_pred)),
    }