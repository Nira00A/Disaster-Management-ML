import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pathlib import Path
import pandas as pd
from typing_extensions import Tuple
from typing_extensions import Optional
import os;
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
import planetary_computer as pc
import numpy as np;
import urllib3
import requests
from rasterio.windows import Window
from tensorflow import keras;
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up 2 levels to reach 'Disaster-Management-AI', then into data/landslide_image_dataset
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "sat_data"))

# Image Boundary
_CATALOG: Optional[Client] = None


catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

BOX_SIZE = 0.0048  

# Image loader function to load the landslide images
def image_loader(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    '''
        The Landslides Images are loaded inside this using Landslide image dataset.    
        The dataset is divided into training and testing sets with a split of 80% for 
        training and 20% for testing.
    '''

    # Loading the dataset from the inbuild directory if not provided
    if data_dir is None:
        data_dir = DATA_DIR

    # Optional: verify it exists before calling keras
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    common_kwargs = {
        "directory": str(data_dir),
        "labels": None,  # No labels; pure feature extraction
        "color_mode": "rgb",
        "batch_size": batch_size,
        "image_size": image_size,
        "shuffle": False,
        "seed": seed,
    }

    # Training Data
    ds = keras.utils.image_dataset_from_directory(**common_kwargs)

    # Returning the training and testing datasets
    return ds

# Converting the dataset to numpy arrays
def dataset_to_numpy(dataset):
    '''
        The dataset is converted to numpy arrays using this function.
    '''

    images, labels = [], []
    for img_batch , img_label in dataset:
        images.append(img_batch.numpy())
        labels.append(img_label.numpy())
    return np.concatenate(images), np.concatenate(labels)

# Loading the dataset to numpy arrays
def load_landslide_numpy_data(
    data_dir: str = DATA_DIR,
    image_size: Tuple[int, int] = (256, 256),
    validation_split: float = 0.2,
    seed: int = 42):
    
    train_ds, test_ds = image_loader(data_dir, image_size, validation_split=validation_split, seed=seed)
    X_train, y_train = dataset_to_numpy(train_ds)
    X_test, y_test = dataset_to_numpy(test_ds)

    return X_train, y_train, X_test, y_test

# Fetching the image from the STAC catalog and for Single Point Data
def fetching_single_image(lat: float, lon: float, date: str):
    """Fetches a 3-channel Sentinel-1 SAR GRD patch using GCP mapping to bypass missing CRS."""
    target_dt = pd.to_datetime(date)
    end_date = target_dt.strftime("%Y-%m-%d")
    start_date = (target_dt - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

    current_bbox = [
        lon - BOX_SIZE,
        lat - BOX_SIZE,
        lon + BOX_SIZE,
        lat + BOX_SIZE,
    ]

    try:
        # 1. Search with fallback window
        stages = [
            f"{start_date}/{end_date}",
            "2019-01-01/2023-12-31",  # Baseline archive window
            None,  # Any scene covering this bbox
        ]

        items = []
        for stage in stages:
            kwargs = {
                "collections": ["sentinel-1-grd"],
                "bbox": current_bbox,
                "max_items": 1,
            }
            if stage:
                kwargs["datetime"] = stage

            search = catalog.search(**kwargs)
            items = list(search.items())
            if items:
                break

        if not items:
            return None

        # 2. Sign item
        item = pc.sign(items[0])
        asset_key = next(
            (
                k
                for k in ["vv", "VV", "vh", "VH"]
                if k in item.assets
            ),
            list(item.assets.keys())[0],
        )

        # 3. Read via GCP mapping (bypasses src.crs = None)
        with rasterio.open(item.assets[asset_key].href) as src:
            gcps, _ = src.gcps
            if gcps:
                gcp_lons = np.array([g.x for g in gcps])
                gcp_lats = np.array([g.y for g in gcps])
                A = np.column_stack(
                    [
                        np.ones(len(gcps)),
                        gcp_lons,
                        gcp_lats,
                        gcp_lons * gcp_lats,
                    ]
                )
                coeff_col, _, _, _ = np.linalg.lstsq(
                    A, np.array([g.col for g in gcps]), rcond=None
                )
                coeff_row, _, _, _ = np.linalg.lstsq(
                    A, np.array([g.row for g in gcps]), rcond=None
                )

                pt = np.array([1.0, lon, lat, lon * lat])
                center_col = int(np.dot(pt, coeff_col))
                center_row = int(np.dot(pt, coeff_row))
            else:
                center_row, center_col = src.index(lon, lat)

            HALF_PX = 50  # ~500m radius window
            win = Window(
                col_off=center_col - HALF_PX,
                row_off=center_row - HALF_PX,
                width=HALF_PX * 2,
                height=HALF_PX * 2,
            )
            arr = src.read(1, window=win, out_shape=(224, 224), boundless=True)

        # 4. Contrast stretch to 0-255 uint8
        valid = arr[arr > 0]
        vmin = np.percentile(valid, 2) if len(valid) > 0 else 0
        vmax = np.percentile(valid, 98) if len(valid) > 0 else 1
        norm = np.clip(
            (arr - vmin) / (vmax - vmin + 1e-6) * 255.0, 0, 255
        ).astype(np.uint8)

        return np.stack([norm, norm, norm], axis=-1)

    except Exception as e:
        print(f"Error fetching SAR image: {e}")
        return None
        
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_landslide_numpy_data()
    # print
    print(f"X_train shape: {X_train}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test}, y_test shape:  {y_test.shape}")
