import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def plot_pca_clusters(X, labels, title, save_path):
    """
    Plot clusters using PCA 2D visualization.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        s=35
    )
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_som_umatrix(som, save_path):
    """
    Plot SOM U-Matrix.
    Higher values indicate larger distances between neighboring neurons.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(som.distance_map().T)
    plt.colorbar(label="Distance")
    plt.title("SOM U-Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()