from pathlib import Path
from typing_extensions import Tuple
from typing_extensions import Optional
import os;
import numpy as np;
from tensorflow import keras;
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up 2 levels to reach 'Disaster-Management-AI', then into data/landslide_image_dataset
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "sat_data"))

# Image loader function to load the landslide images
def image_loader(
    data_dir: str = DATA_DIR,
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
        The dataset is first shuffled and then converted to numpy arrays.
    '''

    images, labels = [], []
    for img_batch , img_label in dataset:
        images.append(img_batch.numpy())
        labels.append(img_label.numpy())
    return np.concatenate(images), np.concatenate(labels)

def load_landslide_numpy_data(
    data_dir: str = DATA_DIR,
    image_size: Tuple[int, int] = (256, 256),
    validation_split: float = 0.2,
    seed: int = 42):
    
    train_ds, test_ds = image_loader(data_dir, image_size, validation_split=validation_split, seed=seed)
    X_train, y_train = dataset_to_numpy(train_ds)
    X_test, y_test = dataset_to_numpy(test_ds)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_landslide_numpy_data()
    # print
    print(f"X_train shape: {X_train}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test}, y_test shape:  {y_test.shape}")
