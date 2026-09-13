import numpy as np
import tensorflow as tf
import pandas as pd
# pyrefly: ignore [missing-import]
from ..data_loaders import image_loader
# pyrefly: ignore [missing-import]
from ..data_loaders import load_landlisde_tabular_data, splitting_data, scale_data
from sklearn.decomposition import PCA
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

## Base Model for Encoding
def encoding(dataDir: str):
    """
    This function encodes the satellite images using ResNet50 model and returns the image 
    embeddings.
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    ## Loading Satellite Images
    img_dataset = image_loader(dataDir)

    img_dataset = img_dataset.map(lambda x: preprocess_input(x))

    image_embeddings = base_model.predict(img_dataset)

    pca = PCA(n_components=256) 

    image_embeddings_reduced = pca.fit_transform(
        image_embeddings
    )

    return image_embeddings_reduced

def join_embeddings_with_tabular_data(image_embeddings, tabular_data):
    """
    This function joins the image embeddings with the tabular data.
    """
    # Ensure that the number of rows in both datasets match
    if image_embeddings.shape[0] != tabular_data.shape[0]:
        raise ValueError("The number of rows in image embeddings and tabular data must match.")

    # Concatenate the image embeddings with the tabular data
    image_df = pd.DataFrame(
        image_embeddings,
        columns=[f"embedding_{i}" for i in range(256)]
    )

    combined_data = pd.concat(
        [image_df, tabular_data.reset_index(drop=True)],
        axis=1
    )

    return combined_data

## Joining the image embeddings with the tabular data

def preprocess_combined_data(combined_data, target_variable: str):
    """
    This function preprocesses the combined data by splitting it into features and target variable,
    and then scaling the features.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = splitting_data(combined_data, target_variable)

    # Scale the features
    columns_to_scale = X_train.columns.tolist()  # Assuming all columns need to be scaled
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, columns_to_scale)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    img_embeddings = encoding()
    combined_data = join_embeddings_with_tabular_data(img_embeddings, load_landlisde_tabular_data())
    preprocessed_data = preprocess_combined_data(combined_data, target_variable="label")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessed_data

    print(f"X_train_scaled shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}, y_test shape: {y_test.shape}")
