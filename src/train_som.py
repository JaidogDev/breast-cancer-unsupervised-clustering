import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans


def _winner_positions_to_array(som, X):
    """
    Convert SOM winner position (row, col) into numeric 2D array.
    """
    return np.array([som.winner(x) for x in X])


def train_som(
    X_train,
    X_test,
    som_x: int = 5,
    som_y: int = 5,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    num_iteration: int = 1000,
    n_clusters: int = 2,
    random_state: int = 42,
):
    """
    Train Kohonen Self-Organizing Map.

    Since SOM produces a 2D map, we convert winning neuron positions into 2 clusters
    using K-Means on SOM winner positions.
    """

    som = MiniSom(
        x=som_x,
        y=som_y,
        input_len=X_train.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=random_state
    )

    som.random_weights_init(X_train)
    som.train_random(X_train, num_iteration=num_iteration)

    train_winners = _winner_positions_to_array(som, X_train)
    test_winners = _winner_positions_to_array(som, X_test)

    # Compress SOM map positions into 2 clusters for binary clustering evaluation.
    som_cluster_model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    train_clusters = som_cluster_model.fit_predict(train_winners)
    test_clusters = som_cluster_model.predict(test_winners)

    return som, som_cluster_model, train_clusters, test_clusters, train_winners, test_winners