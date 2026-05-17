from sklearn.cluster import KMeans


def train_kmeans(X_train, X_test, n_clusters: int = 2, random_state: int = 42):
    """
    Train K-Means using only feature data.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    train_clusters = model.fit_predict(X_train)
    test_clusters = model.predict(X_test)

    return model, train_clusters, test_clusters