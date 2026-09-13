import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pathlib import Path
import pandas as pd
from typing_extensions import Tuple
from typing_extensions import Optional
import os;
from pystac_client import Client
import planetary_computer as pc
import numpy as np;
from tensorflow import keras;
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up 2 levels to reach 'Disaster-Management-AI', then into data/landslide_image_dataset
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "sat_data"))

# Image Boundary
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

BOX_SIZE = 0.0048  # ~2.5 km box

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
    """
    Fetches the image from the STAC catalog for the given latitude, longitude, and date.
    """
    dt = pd.to_datetime(date)
    year = dt.year
    start_date = dt - pd.Timedelta(days=7)

    # Format as plain YYYY-MM-DD — STAC API rejects datetime strings with time components
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = dt.strftime("%Y-%m-%d")

    current_bbox = [lon - BOX_SIZE, lat - BOX_SIZE, lon + BOX_SIZE, lat + BOX_SIZE]

    try:
        # Sort by cloud cover over the 4-month dry window to grab the clearest scene (<5% cloud cover)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=current_bbox,
            datetime=f"{start_date_str}/{end_date_str}",
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            max_items=1
        )
        items = list(search.items())
        
        if not items:
            fallback_start = f"{year - 1}-01-01"
            fallback_end = f"{year - 1}-04-30"
            search_fallback = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=current_bbox,
                datetime=f"{fallback_start}/{fallback_end}",
                sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
                max_items=1
            )
            items = list(search_fallback.items())
        
        if not items:
            return None

        item = pc.sign(items[0])
        visual_url = item.assets["visual"].href

        # Reproject, stream, and crop 224x224 RGB image
        with rasterio.open(visual_url) as src:
            proj_bbox = transform_bounds("EPSG:4326", src.crs, *current_bbox)
            window = from_bounds(*proj_bbox, transform=src.transform)
            rgb = src.read([1, 2, 3], window=window, out_shape=(3, 224, 224), boundless=True)
            rgb = np.moveaxis(rgb, 0, -1)

        return rgb

    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_landslide_numpy_data()
    # print
    print(f"X_train shape: {X_train}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test}, y_test shape:  {y_test.shape}")
