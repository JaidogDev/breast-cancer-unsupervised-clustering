# Unsupervised Breast Cancer Clustering: K-Means vs Kohonen SOM

## Overview

This project applies unsupervised machine learning techniques to the Breast Cancer Wisconsin Diagnostic dataset.

The goal is to compare:

- K-Means Clustering
- Kohonen Self-Organizing Map (SOM)

for clustering breast cancer samples using only numeric feature values.

Although the dataset contains true labels:

- malignant
- benign

these labels are **not used during model training**.  
They are used only after clustering to evaluate how well the discovered clusters align with the actual classes.

This project is therefore an:

```text
Unsupervised clustering task with external label evaluation
```

## Dataset

Dataset source: Scikit-learn Breast Cancer Wisconsin Diagnostic Dataset  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

The dataset contains numeric features computed from digitized images of breast mass cell nuclei.

Dataset information:

- Dataset: Breast Cancer Wisconsin Diagnostic
- Samples: 569
- Features: 30 numeric features
- Classes: malignant, benign
- Train/Test split: 80/20
- Train samples: 455
- Test samples: 114

The original dataset comes from the UCI Machine Learning Repository and was donated in 1995.

## Methods

### K-Means Clustering

K-Means is a centroid-based clustering algorithm.

It tries to divide data points into clusters by finding cluster centers, or centroids.  
Each data point is assigned to the nearest centroid.

In this project:

```text
n_clusters = 2
```

because the dataset contains two actual classes: malignant and benign.

### Kohonen Self-Organizing Map

Kohonen Self-Organizing Map, or SOM, is an unsupervised neural network technique.

SOM maps high-dimensional input data into a 2D neuron grid.  
Each input sample activates the closest neuron, called the Best Matching Unit (BMU).

Unlike K-Means, SOM also tries to preserve the topology of the data, meaning similar samples should be mapped to nearby neurons on the 2D map.

## Slight Code Change

To demonstrate the effect of SOM map resolution, this project slightly changes the SOM map size.

The experiments compare:

- SOM 5x5
- SOM 10x10
- SOM 20x20

The number of neurons increases as follows:

| SOM Map Size | Number of Neurons |
|---|---:|
| 5x5 | 25 |
| 10x10 | 100 |
| 20x20 | 400 |

The purpose is to study whether increasing the SOM map size improves clustering performance.

## Evaluation Metrics

The models are evaluated using clustering metrics:

- Silhouette Score
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Purity

Silhouette Score evaluates cluster separation without using true labels.

ARI, NMI, and Purity use the true labels only after clustering to measure how well the discovered clusters align with malignant/benign labels.

## Results

| Model | Silhouette | ARI | NMI | Purity |
|---|---:|---:|---:|---:|
| K-Means | 0.3268 | 0.7354 | 0.6188 | 0.9298 |
| SOM 5x5 | 0.1096 | 0.2179 | 0.1944 | 0.7368 |
| SOM 10x10 | 0.0230 | -0.0102 | 0.0018 | 0.6316 |
| SOM 20x20 | -0.0008 | 0.0261 | 0.0573 | 0.6316 |

K-Means achieved the best performance across all metrics.

Among the SOM experiments, the 5x5 map performed better than the larger 10x10 and 20x20 maps.

## Visualization Results

### 1. True Labels PCA Plot

![True Labels PCA Plot](outputs/som_5x5/true_labels_pca_plot.png)

This plot shows the actual malignant and benign labels projected into 2D space using PCA.  
It is used as a reference visualization to compare how well each clustering method aligns with the true labels.

### 2. K-Means Clustering Result

![K-Means PCA Plot](outputs/som_5x5/kmeans_pca_plot.png)

This plot shows the clusters produced by K-Means.  
Compared with the true labels, K-Means produces a clearer separation between the two groups, which matches its higher ARI, NMI, and Purity scores.

### 3. Kohonen SOM Clustering Result

![SOM PCA Plot](outputs/som_5x5/som_pca_plot.png)

This plot shows the clustering result from the Kohonen SOM-based pipeline.  
The SOM clusters are more mixed compared with K-Means, which is consistent with the lower clustering metrics.

### 4. SOM U-Matrix

![SOM U-Matrix](outputs/som_5x5/som_umatrix.png)

The U-Matrix visualizes distances between neighboring neurons in the SOM map.  
Regions with larger distances can indicate possible boundaries between groups, making SOM useful for exploring the topology of high-dimensional data.

## Result Interpretation

The results show that increasing the SOM map size does not always improve clustering performance.

At first, a larger SOM map may seem better because it provides more neurons and a more detailed 2D representation. However, this dataset has only 569 samples and 455 training samples.

When the SOM map becomes too large, such as 20x20 with 400 neurons, the number of neurons becomes too high compared with the number of training samples. This can cause the data to spread too sparsely across the map, making the final cluster structure less clear.

This behavior can be interpreted as over-segmentation.

In this experiment:

- SOM 5x5 performs best among SOM settings
- SOM 10x10 and 20x20 perform worse
- K-Means still outperforms SOM overall

This suggests that K-Means is more suitable for this dataset in terms of clustering performance, while SOM is still useful for visualization and understanding the data topology through the SOM U-Matrix.

## How to Run

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run experiment
python main.py
```

## Project Structure

```text
ml_breast_cancer_kmean_kohonen/
├── data/
│   └── breast_cancer.csv
├── outputs/
│   ├── som_5x5/
│   ├── som_10x10/
│   └── som_20x20/
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── plot_utils.py
│   ├── prepare_data.py
│   ├── train_kmeans.py
│   └── train_som.py
├── main.py
├── requirements.txt
└── README.md
```

## Experiment Summary

| Experiment | Main Change | Observation |
|---|---|---|
| K-Means | Centroid-based clustering with 2 clusters | Best overall clustering performance |
| SOM 5x5 | 25-neuron SOM map | Best SOM result |
| SOM 10x10 | Increased map size to 100 neurons | Performance decreased |
| SOM 20x20 | Increased map size to 400 neurons | Performance did not improve |

## Conclusion

This project demonstrates the difference between centroid-based clustering and topology-preserving clustering.

K-Means performs better on the Breast Cancer dataset in terms of clustering metrics.  
SOM provides useful visualization through the 2D neuron map and U-Matrix, but its performance depends strongly on the selected map size.

The experiment shows that increasing SOM map size does not guarantee better clustering performance. For this dataset, a smaller 5x5 SOM map is more suitable than larger 10x10 or 20x20 maps.

Overall, K-Means is more effective for clustering this dataset, while SOM is useful for exploring and visualizing the structure of high-dimensional data.

## Example Metrics Output

A full metrics file is saved automatically after running the experiment:

```text
outputs/som_5x5/metrics.json
outputs/som_10x10/metrics.json
outputs/som_20x20/metrics.json
```

```json
{
  "kmeans": {
    "silhouette_score": 0.3268,
    "adjusted_rand_index": 0.7354,
    "normalized_mutual_info": 0.6188,
    "purity": 0.9298
  },
  "kohonen_som_5x5": {
    "silhouette_score": 0.1096,
    "adjusted_rand_index": 0.2179,
    "normalized_mutual_info": 0.1944,
    "purity": 0.7368
  }
}
```
