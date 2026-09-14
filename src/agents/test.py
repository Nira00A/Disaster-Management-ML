from pathlib import Path
# pyrefly: ignore [missing-import]
from ..ebsemble import EnsembleModelPipeline

BASE_PATH = Path("./models")

# 1. Initialize and load artifacts
pipeline = EnsembleModelPipeline.load("ensemble_model_revised.joblib")

# 2. Predict on new input (image path, PIL Image, or numpy array)

prediction = pipeline.predict(26.3648598541958,94.0925822794819,"17-03-2024")
probabilities = pipeline.predict_proba(26.3648598541958,94.0925822794819,"17-03-2024")

print(f"Prediction: {prediction}")
print(f"Landslide Risk Probability: {probabilities}")