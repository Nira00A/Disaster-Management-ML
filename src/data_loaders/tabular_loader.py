from pandas import DataFrame
from pathlib import Path
import pandas as pd
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple

## Data Directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
'''DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "final_tabular_dataset.csv"))'''
DATA_DIR_LAT_LON = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "final_with_lat_lon.csv"))

## Loading the dataset of the Landslides Containing feilds like - id , latitude , longitude , geometry .....
def load_landlisde_tabular_data(
    data_dir: str):

    # Fallback directory
    if data_dir is None:
        data_dir = DATA_DIR_LAT_LON

    # Load dataset
    df = pd.read_csv(data_dir)

    print(df.shape)
    return df

## Train test split

def add_kmeans_spatial_clusters(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    n_clusters: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Creates spatial clusters from coordinates using K-Means."""
    df_clustered = df.copy()
    coords = df_clustered[[lat_col, lon_col]].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_clustered["spatial_cluster"] = kmeans.fit_predict(coords)

    return df_clustered

def splitting_data(
    df: pd.DataFrame,
    target_variable: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    n_clusters: int = 10,
    test_size: float = 0.4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Clusters coordinates into zones and performs a spatial holdout split.
    
    Uses the df passed in (which may contain image embeddings + tabular features)
    for both clustering and training. lat/lon columns are used for spatial
    grouping only and are dropped from the final feature matrix X.
    """
    df_clustered = add_kmeans_spatial_clusters(
        df, lat_col=lat_col, lon_col=lon_col, n_clusters=n_clusters, random_state=random_state
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df_clustered["spatial_cluster"]

    train_idx, test_idx = next(gss.split(df_clustered, groups=groups))

    # Drop target + spatial helpers. Only drop columns that actually exist in df.
    cols_to_drop = [c for c in [target_variable, "spatial_cluster", lat_col, lon_col]
                    if c in df_clustered.columns]
    X = df_clustered.drop(columns=cols_to_drop)
    y = df_clustered[target_variable]

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

## Scaling the data
def scale_data(Xtrain , Xtest , columns: list[str]):
    scale = StandardScaler()
    Xtrain_scaled = Xtrain.copy()
    Xtrain_scaled[columns] = scale.fit_transform(Xtrain[columns])

    Xtest_scaled = Xtest.copy()
    Xtest_scaled[columns] = scale.transform(Xtest[columns])
    return Xtrain_scaled, Xtest_scaled

if __name__ == "__main__":
    df = load_landlisde_tabular_data()
    print(df.head())
    '''Xtrain, Xtest, ytrain, ytest = splitting_data(df, "label")
    Xtrain_scaled, Xtest_scaled = scale_data(Xtrain, Xtest, ["soil_moisture_0_7cm_avg", "soil_moisture_7_28cm_avg", "daily_rainfall_mm", "cumulative_rainfall_3d_mm", "cumulative_rainfall_7d_mm", "elev_1", "slope_1", "rough_1", "tri_1"])
    print(Xtrain_scaled.head())
    print(Xtest_scaled.head())'''