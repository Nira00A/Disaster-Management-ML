import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
# pyrefly: ignore [missing-import]
from ..fusion_models import encoding, join_embeddings_with_tabular_data, preprocess_combined_data
# pyrefly: ignore [missing-import]
from ..data_loaders import image_loader, load_landlisde_tabular_data, fetching_single_image, get_topology, get_terrain_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
import tensorflow as tf

# Model Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = PROJECT_ROOT / "models"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAT_DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "sat_data"))
TABULAR_DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "final_tabular_dataset.csv"))

'''print(X_train.info())
print(X_train.isnull().sum())
print(X_test.info())
print(y_train.info())
print(y_test.info())

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)'''

class ImageFeatureExtractor(BaseEstimator):
    def __init__(self,pca=None, model=None):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        self.pca = pca if pca is not None else PCA(n_components=256)

    def _train_image_features(self):
        """
        Load all training satellite images, run them through self.model (ResNet50),
        and fit self.pca on the resulting embeddings.
        Returns the PCA-reduced embeddings for downstream training.
        """

        # Load entire sat_data directory as an unlabelled tf.data.Dataset
        img_dataset = image_loader(SAT_DATA_DIR)
        img_dataset = img_dataset.map(lambda x: preprocess_input(x))

        # Get raw ResNet50 embeddings (global average pooled — shape: N x 2048)
        self.model.trainable = False
        raw_embeddings = self.model.predict(img_dataset)

        # Fit self.pca HERE so that transform() works correctly at prediction time
        img_embeddings_reduced = self.pca.fit_transform(raw_embeddings)

        return img_embeddings_reduced

    def _extract_image_features(self, lat, lon, date):
        revised_date = pd.to_datetime(date)

        ## Fetch the raw numpy RGB image from STAC
        img_array = fetching_single_image(lat, lon, revised_date)

        if img_array is None:
            print("Warning: image fetch failed — returning zero embeddings.")
            return np.zeros((1, 256))

        # img_array is (224, 224, 3) numpy array — add batch dim and wrap in tf.data.Dataset
        img_tensor = tf.cast(img_array[np.newaxis, ...], tf.float32)  # shape: (1, 224, 224, 3)
        img_dataset = tf.data.Dataset.from_tensors(img_tensor).map(
            lambda x: preprocess_input(x)
        )

        image_embeddings = self.model.predict(img_dataset)

        image_embeddings_reduced = self.pca.transform(
            image_embeddings
        )

        return image_embeddings_reduced

class EnsembleModelPipeline(BaseEstimator):
    ## Initializing the Model
    def __init__(self, ensemble_model=None, image_extractor=None, scaler=None):
        self.ensemble_model = self.base_estimators() if ensemble_model is None else ensemble_model
        # Use the provided extractor (e.g. loaded from disk with fitted PCA) or create a fresh one
        self.image_extractor = image_extractor if image_extractor is not None else ImageFeatureExtractor()
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.accuracy = None
        self.classification_report = None
        self.confusion_matrix = None

    ## Defining the Base Estimators (Random Forest, SVC, ANN)
    def base_estimators(self):
        rf_model = RandomForestClassifier(
            n_estimators=500,
            criterion="entropy",
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        svm_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("svm", SVC(
            kernel="rbf",
            gamma="auto",
            class_weight="balanced",
            probability=True,
            random_state=42
            ))
        ])

        ann_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("ann", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                activation="relu",
                solver="adam",
                random_state=42,
                learning_rate_init=0.001,
            ))
        ])

        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('svm', svm_pipe),
                ('ann', ann_pipe),
            ],
            voting='soft',
            weights=[1, 1, 1],
        )

        return ensemble_model

    ## Preprocessing the combined data
    def preprocess(self):
        ## Extracting Image Features
        img_embeddings = self.image_extractor._train_image_features()

        combined_data = join_embeddings_with_tabular_data(img_embeddings, load_landlisde_tabular_data(TABULAR_DATA_DIR))

        ## Preprocessing the combined data
        X_train, X_test, y_train, y_test = preprocess_combined_data(combined_data, target_variable="label")

        ## Scaling the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    ## Feature Extraction
    def _extracting_features(self, lat, lon, date):
        image_embeddings_reduced = self.image_extractor._extract_image_features(lat, lon, date)
        
        # Extracting the corresponding values for each features in the tabular data based on the lat and lon
        topology_features = get_topology(lat, lon, date)
        terrain_features = get_terrain_features(lat, lon)

        df_image = pd.DataFrame(
            [image_embeddings_reduced.flatten()],
            columns=[f"embedding_{i}" for i in range(256)],
        )
        df_terrain = pd.DataFrame([terrain_features])
        df_topology = pd.DataFrame([topology_features])

        # Concatenate them side-by-side
        combined_features = pd.concat([df_image,df_topology, df_terrain], axis=1)

        # Scale the combined features
        combined_features_scaled = self.scaler.transform(combined_features)

        return combined_features_scaled

    ## Fitting the Model
    def fit(self, X_train=None, y_train=None):
        if X_train is None and y_train is None:
            X_train, X_test, y_train, y_test = self.preprocess()

        # Fit the ensemble model
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate the ensemble model
        self.accuracy = accuracy_score(y_test, self.ensemble_model.predict(X_test))
        self.classification_report = classification_report(y_test, self.ensemble_model.predict(X_test))
        self.confusion_matrix = confusion_matrix(y_test, self.ensemble_model.predict(X_test))

    ## Prediction probability
    def predict_proba(self, latitude: float, longitude: float, date: str):
        features = self._extracting_features(latitude, longitude, date)
        proba = self.ensemble_model.predict_proba(features)[0]  # shape: (n_classes,)

        classes = list(self.ensemble_model.classes_) 

        # Probability of the landslide class (landslide = 0, not landslide = 1)
        landslide_prob = proba[classes.index(0)] if 0 in classes else proba[1]
        landslide_chances = round(float(landslide_prob) * 100, 2)

        return landslide_chances
        
    ## Prediction
    def predict(self, latitude:float, longitude:float, date:str):
        features = self._extracting_features(latitude, longitude, date)

        # 2. Predict directly from the pre-processed training vector
        print("Classes order:", self.ensemble_model.classes_)
        print("Direct X_train proba:", self.ensemble_model.predict_proba(features))
        print("Direct X_train pred:", self.ensemble_model.predict(features))
        result = self.ensemble_model.predict(features)
        return result

    ## Exporting the model
    def export(self, filename="ensemble_artifacts.joblib"):
        """Saves VotingClassifier and fitted PCA together."""
        artifacts = {
            "ensemble_model": self.ensemble_model,
            "pca": self.image_extractor.pca,
            "scaler": self.scaler
        }
        export_path = BASE_PATH / filename
        joblib.dump(artifacts, export_path)
        print(f"Artifacts successfully exported to: {export_path}")

    ## Displaying the metrics
    def display_metrics(self):
        print("Accuracy Score:", self.accuracy)
        print("Classification Report:", self.classification_report)
        print("Confusion Matrix:\n", self.confusion_matrix)

    @classmethod
    def load(cls, filename="landslide_model.joblib"):
        """Loads artifacts and returns ready-to-use pipeline."""
        artifacts = joblib.load(BASE_PATH / filename)

        extractor = ImageFeatureExtractor(pca=artifacts["pca"])
        return cls(ensemble_model=artifacts["ensemble_model"], image_extractor=extractor, scaler=artifacts["scaler"])

if __name__ == "__main__":
    #pipeline = EnsembleModelPipeline.load("ensemble_model_revised.joblib")
    pipeline = EnsembleModelPipeline()
    pipeline.fit()
    prediction = pipeline.predict(24.780027,92.45372,"02-06-2020")
    print(pipeline.predict_proba(24.780027,92.45372,"02-06-2020"))
    pipeline.export("ensemble_model_revised.joblib")
