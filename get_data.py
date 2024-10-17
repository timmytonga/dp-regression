import numpy as np
import pandas as pd
from openml import tasks

def get_data_from_id(id):
    dataset = tasks.get_task(id).get_dataset()

    # Fetch features and labels
    X, Y, categorical, feature_names = dataset.get_data(target=dataset.default_target_attribute)

    # Convert to DataFrame
    data = pd.DataFrame(X)
    data.columns = feature_names
    n, d = X.shape

    # Add labels to data
    data.insert(d, "label", Y)

    # Convert categorical features to DataFrame
    categorical_df = pd.DataFrame(categorical)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Extract labels (Y)
    Y = data.iloc[:, -1].to_numpy()

    # Remove labels from the data
    data = data.iloc[:, :-1]

    # Identify categorical columns
    categorical_indices = np.nonzero(categorical_df.to_numpy())[0]

    # One-hot encode categorical columns
    for category_idx in categorical_indices:
        expanded = pd.get_dummies(data[data.columns[category_idx]])
        data = pd.concat([data, expanded], axis=1)

    # Drop the original categorical columns that were expanded
    data = data.drop(data.columns[categorical_indices], axis=1)

    # Add intercept feature (a column of ones)
    X = data.to_numpy()
    X = np.column_stack((X, np.ones(X.shape[0]))).astype(float)

    # Slightly perturb the data to avoid ties for Kendall
    X = X + np.random.uniform(low=-1e-6, high=1e-6, size=X.shape)
    Y = Y + np.random.uniform(low=-1e-6, high=1e-6, size=Y.shape)

    return X, Y
