import os
import json
import pandas as pd
import numpy as np

from src.prepare_data import load_and_prepare_data
from src.train_kmeans import train_kmeans
from src.train_som import train_som
from src.evaluate import evaluate_clustering
from src.plot_utils import plot_pca_clusters, plot_som_umatrix


RANDOM_STATE = 42

# =========================
# SOM experiment settings
# Change only these values to run another SOM map size
# Example: 5x5, 10x10, 20x20
# =========================
SOM_X = 20
SOM_Y = 20

OUTPUT_DIR = f"outputs/som_{SOM_X}x{SOM_Y}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================
    # 1. Load and prepare data
    # =========================
    data = load_and_prepare_data(
        output_csv_path="data/breast_cancer.csv",
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    target_names = data["target_names"]

    print("Dataset: Breast Cancer Wisconsin Diagnostic")
    print("Dataset shape:", data["dataset_shape"])
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Target names:", target_names)
    print("Saved dataset CSV to:", data["csv_path"])
    print(f"SOM map size: {SOM_X}x{SOM_Y}")
    print(f"Output directory: {OUTPUT_DIR}")

    # =========================
    # 2. Train K-Means
    # =========================
    kmeans_model, kmeans_train_clusters, kmeans_test_clusters = train_kmeans(
        X_train=X_train,
        X_test=X_test,
        n_clusters=2,
        random_state=RANDOM_STATE
    )

    # =========================
    # 3. Train Kohonen SOM
    # =========================
    (
        som,
        som_cluster_model,
        som_train_clusters,
        som_test_clusters,
        som_train_winners,
        som_test_winners
    ) = train_som(
        X_train=X_train,
        X_test=X_test,
        som_x=SOM_X,
        som_y=SOM_Y,
        sigma=1.0,
        learning_rate=0.5,
        num_iteration=1000,
        n_clusters=2,
        random_state=RANDOM_STATE
    )

    # =========================
    # 4. Evaluate on test set
    # =========================
    kmeans_metrics = evaluate_clustering(
        X=X_test,
        y_true=y_test,
        y_pred=kmeans_test_clusters
    )

    som_metrics = evaluate_clustering(
        X=X_test,
        y_true=y_test,
        y_pred=som_test_clusters
    )

    results = {
        "project": "Breast Cancer Clustering: K-Means vs Kohonen SOM",
        "dataset": {
            "name": "Breast Cancer Wisconsin Diagnostic",
            "source": "sklearn.datasets.load_breast_cancer",
            "instances": int(data["dataset_shape"][0]),
            "features": int(data["dataset_shape"][1]),
            "classes": list(target_names),
            "note": "Labels are not used during clustering. Labels are used only for evaluation."
        },
        "split": {
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "test_size": 0.2,
            "random_state": RANDOM_STATE
        },
        "models": {
            "kmeans": {
                "n_clusters": 2,
                "metrics": kmeans_metrics
            },
            "kohonen_som": {
                "map_size": f"{SOM_X}x{SOM_Y}",
                "sigma": 1.0,
                "learning_rate": 0.5,
                "num_iteration": 1000,
                "metrics": som_metrics
            }
        }
    }

    print("\nEvaluation Results")
    print("==================")
    print(json.dumps(results, indent=4))

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # =========================
    # 5. Save cluster report
    # =========================
    report_df = pd.DataFrame({
        "true_label": y_test,
        "true_label_name": [target_names[i] for i in y_test],
        "kmeans_cluster": kmeans_test_clusters,
        "som_cluster": som_test_clusters
    })

    report_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_report.csv"), index=False)

    # =========================
    # 6. Visualizations
    # =========================
    plot_pca_clusters(
        X=X_test,
        labels=y_test,
        title="True Labels on Test Set",
        save_path=os.path.join(OUTPUT_DIR, "true_labels_pca_plot.png")
    )

    plot_pca_clusters(
        X=X_test,
        labels=kmeans_test_clusters,
        title="K-Means Clusters on Test Set",
        save_path=os.path.join(OUTPUT_DIR, "kmeans_pca_plot.png")
    )

    plot_pca_clusters(
        X=X_test,
        labels=som_test_clusters,
        title=f"Kohonen SOM Clusters on Test Set ({SOM_X}x{SOM_Y})",
        save_path=os.path.join(OUTPUT_DIR, "som_pca_plot.png")
    )

    plot_som_umatrix(
        som=som,
        save_path=os.path.join(OUTPUT_DIR, "som_umatrix.png")
    )

    print("\nSaved outputs to:", OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()