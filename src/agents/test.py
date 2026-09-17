import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import pandas as pd
from pystac_client import Client
import planetary_computer as pc
import rasterio
from rasterio.windows import Window
import requests
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

load_dotenv()

# Path for models
BASE_PATH = Path("./models")
BOX_SIZE = 0.0048 #Size of the box to be extracted from the satellite image

# Initialize client 
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


class LandslideModelPipeline:
    '''
        This is the class of Landslide Model Pipeline which will be used to predict the probability of landslide
        at a given latitude and longitude and date.
    '''
    def __init__(self, repo_id: str | None = None):
        # Load Ensemble Model Artifacts
        self.scaler = None
        self.pca = None
        self.model = None
        self.threshold = 0.40

        # ResNet50 Model
        self.resnet = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        self.resnet.trainable = False

        if repo_id:
            self.load_model(repo_id)

    def load_model(
        self,
        repo_id: str,
        filename: str = "ensemble_artifactsV2.joblib",
    ):
        """Downloads and unpacks model weights, fitted PCA, scaler, and threshold."""
        hf_token = os.getenv("HF_TOKEN")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token,
            cache_dir=BASE_PATH,
        )

        artifacts = joblib.load(model_path)

        self.scaler = artifacts["scaler"]
        self.pca = artifacts["pca"]
        self.model = artifacts["ensemble_model"]
        self.threshold = artifacts.get("threshold", 0.40)

    def predict(self, lat: float, lon: float, date: str, repo_id: str | None = None):
        """Orchestrates feature extraction and returns probability and binary classification."""
        if self.model is None:
            if repo_id is None:
                raise ValueError("Model not loaded and no `repo_id` provided.")
            self.load_model(repo_id)

        combined_features_scaled = self._extract_features(lat, lon, date)

        # Predict probability of Class 1 (Landslide)
        proba_array = self.model.predict_proba(combined_features_scaled)[0]
        classes = list(getattr(self.model, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else 1

        landslide_probability = round(float(proba_array[pos_idx]) * 100, 2)
        prediction_label = 1 if (landslide_probability / 100.0) >= self.threshold else 0

        return landslide_probability, prediction_label

    def _extract_features(self, lat: float, lon: float, date: str) -> np.ndarray:
        image_embeddings_reduced = self._extract_image_features(lat, lon, date)
        topology_features = self._get_topology(lat, lon, date)
        terrain_features = self._get_terrain_features(lat, lon)

        df_image = pd.DataFrame(
            [image_embeddings_reduced.flatten()],
            columns=[f"embedding_{i}" for i in range(16)],
        )
        df_topology = pd.DataFrame([topology_features])
        df_terrain = pd.DataFrame([terrain_features])

        # Concatenate matching training feature schema
        combined_features = pd.concat([df_image, df_topology, df_terrain], axis=1)
        return self.scaler.transform(combined_features)

    def _get_terrain_features(
        self, lat: float, lon: float, step: float = 0.000278
    ) -> dict:
        """Extracts elevation, slope, roughness, and TRI via Open-Meteo API."""
        lats = [
            lat + step, lat + step, lat + step,
            lat, lat, lat,
            lat - step, lat - step, lat - step,
        ]
        lons = [
            lon - step, lon, lon + step,
            lon - step, lon, lon + step,
            lon - step, lon, lon + step,
        ]

        url = "https://api.open-meteo.com/v1/elevation"
        params = {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
        }

        try:
            res = requests.get(url, params=params, timeout=5)
            res.raise_for_status()
            elevations = res.json().get("elevation", [])

            if len(elevations) != 9:
                raise ValueError("Incomplete elevation response.")

            data = np.array(elevations, dtype=float).reshape((3, 3))
            z0 = data[1, 1]

            dx = step * (111320 * np.cos(np.radians(lat)))
            dy = step * 110540

            z1, z2, z3 = data[0, 0], data[0, 1], data[0, 2]
            z4, _, z5 = data[1, 0], data[1, 1], data[1, 2]
            z6, z7, z8 = data[2, 0], data[2, 1], data[2, 2]

            dz_dx = ((z3 + 2 * z5 + z8) - (z1 + 2 * z4 + z6)) / (8 * dx)
            dz_dy = ((z6 + 2 * z7 + z8) - (z1 + 2 * z2 + z3)) / (8 * dy)
            slope = float(np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))))

            return {
                "elevation": round(float(z0), 2),
                "slope": round(slope, 4),
                "roughness": round(float(np.max(data) - np.min(data)), 4),
                "tri": round(float(np.sqrt(np.sum((data - z0) ** 2))), 4),
            }
        except Exception as e:
            print(f"Warning: Terrain API query failed ({e}) — using fallback baseline.")
            return {"elevation": 420.0, "slope": 18.5, "roughness": 22.0, "tri": 28.5}

    def _get_topology(self, lat: float, lon: float, target_date: str) -> dict:
        """Fetches soil moisture and antecedent rainfall from Open-Meteo Historical Archive."""
        target_dt = pd.to_datetime(target_date).tz_localize("UTC")
        start_dt = target_dt - pd.Timedelta(days=7)

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": target_dt.strftime("%Y-%m-%d"),
            "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,precipitation",
            "timezone": "UTC",
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data or "hourly" not in data:
                raise ValueError("Missing hourly telemetry.")

            hourly = data["hourly"]
            df = pd.DataFrame(
                {
                    "time": pd.to_datetime(hourly["time"], utc=True),
                    "soil_0_7": hourly["soil_moisture_0_to_7cm"],
                    "soil_7_28": hourly["soil_moisture_7_to_28cm"],
                    "precip": hourly["precipitation"],
                }
            )

            target_mask = df["time"].dt.date == target_dt.date()
            past_3d_mask = df["time"] >= (target_dt - pd.Timedelta(days=3))

            return {
                "soil_moisture_0_7cm_avg": float(df.loc[target_mask, "soil_0_7"].mean()),
                "soil_moisture_7_28cm_avg": float(df.loc[target_mask, "soil_7_28"].mean()),
                "daily_rainfall_mm": float(df.loc[target_mask, "precip"].sum()),
                "cumulative_rainfall_3d_mm": float(df.loc[past_3d_mask, "precip"].sum()),
                "cumulative_rainfall_7d_mm": float(df["precip"].sum()),
            }
        except Exception as e:
            print(f"Warning: Weather API query failed ({e}) — using zeroed baseline.")
            return {
                "soil_moisture_0_7cm_avg": 0.0,
                "soil_moisture_7_28cm_avg": 0.0,
                "daily_rainfall_mm": 0.0,
                "cumulative_rainfall_3d_mm": 0.0,
                "cumulative_rainfall_7d_mm": 0.0,
            }

    def _fetching_single_image(self, lat: float, lon: float, date: str) -> np.ndarray | None:
        """Fetches a Sentinel-1 SAR patch using GCP mapping."""
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
            stages = [
                f"{start_date}/{end_date}",
                "2019-01-01/2023-12-31",
                None,
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

            item = pc.sign(items[0])
            asset_key = next(
                (k for k in ["vv", "VV", "vh", "VH"] if k in item.assets),
                list(item.assets.keys())[0],
            )

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

                half_px = 50
                win = Window(
                    col_off=center_col - half_px,
                    row_off=center_row - half_px,
                    width=half_px * 2,
                    height=half_px * 2,
                )
                arr = src.read(1, window=win, out_shape=(224, 224), boundless=True)

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

    def _extract_image_features(self, lat: float, lon: float, date: str) -> np.ndarray:
        img_array = self._fetching_single_image(lat, lon, date)

        if img_array is None:
            print("Warning: image fetch failed — returning zero embeddings.")
            return np.zeros((1, 16))

        img_tensor = tf.cast(img_array[np.newaxis, ...], tf.float32)
        img_preprocessed = preprocess_input(img_tensor)

        raw_embeddings = self.resnet.predict(img_preprocessed, verbose=0)
        return self.pca.transform(raw_embeddings)


if __name__ == "__main__":
    '''
    Testing the model
    '''
    pipeline = LandslideModelPipeline(repo_id="Nira00A/landslide-early-warning-ensemble")

    lat = 24.780027
    lon = 92.45372
    date = "2020-06-02"

    probability, label = pipeline.predict(lat, lon, date)

    print(f"Prediction Probability: {probability}%")
    print(f"Prediction Label: {label} ({'CRITICAL HAZARD' if label == 1 else 'SAFE'})")