from pathlib import Path
# pyrefly: ignore [missing-import]
from ..ebsemble import EnsembleModelPipeline

BASE_PATH = Path("./models")

# 1. Initialize and load artifacts
pipeline = EnsembleModelPipeline.load("ensemble_model_revised.joblib")

# 2. Predict on new input (image path, PIL Image, or numpy array)

prediction = pipeline.predict(25.609131,93.258236,"2000-09-01")
probabilities = pipeline.predict_proba(25.609131,93.258236,"2000-09-01")

print(f"Prediction: {prediction[0]}")
print(f"Landslide Risk Probability: {probabilities}")