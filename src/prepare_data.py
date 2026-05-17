import os
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(
    output_csv_path: str = "data/breast_cancer.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load Breast Cancer Wisconsin dataset.

    Important:
    - Features are used for unsupervised clustering.
    - Labels are NOT used for training K-Means or SOM.
    - Labels are used only for evaluation after clustering.
    """

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    data = load_breast_cancer()

    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df["target_name"] = [target_names[i] for i in y]
    df.to_csv(output_csv_path, index=False)

    # Split for evaluation.
    # y is used only for stratified split and evaluation, not for model training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "target_names": target_names,
        "scaler": scaler,
        "dataset_shape": X.shape,
        "csv_path": output_csv_path
    }