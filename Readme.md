# Disaster-Management-ML

Production-ready machine learning pipeline for landslide risk prediction using multimodal data (satellite imagery + tabular geospatial and weather features).

## V1 Model: How It Works

### Workflow Diagram

![V1 Model Workflow](https://github.com/user-attachments/assets/b29b38c6-cdfd-49ff-8220-a49c6faa0426)

![V1 Prediction Flow](https://github.com/user-attachments/assets/955dd697-b6ba-48f5-bcc2-496126b482de)

### 1) Inputs

V1 combines two data streams for each location sample:

- **Satellite images** (loaded as 224×224 RGB tensors)
- **Structured tabular features** (topology, rainfall, and soil-related attributes)

### 2) Image Feature Extraction

- A pretrained **ResNet50** backbone (`include_top=False`, `pooling="avg"`) converts each image into a dense embedding.
- Embeddings are then compressed using **PCA to 256 components** to retain useful variance while reducing model complexity.

### 3) Feature Fusion

- Reduced image embeddings are converted into tabular columns (`embedding_0 ... embedding_255`).
- These columns are concatenated with the structured tabular dataset to build one unified training matrix.

### 4) Preprocessing

- The fused dataset is split into train/test sets.
- Numerical feature scaling is applied using `StandardScaler`.
- Missing values inside SVM/ANN branches are handled with `SimpleImputer(strategy="mean")`.

### 5) Ensemble Learning (V1)

V1 trains three base models and combines them with **soft voting**:

1. **Random Forest** (`n_estimators=500`, entropy criterion, balanced class weights)
2. **SVM (RBF kernel)** with probability outputs enabled
3. **ANN (MLPClassifier)** with hidden layers `(128, 64)`

The final classifier is a `VotingClassifier` with equal weights across all three models.

### 6) Prediction Generation

For any new sample, prediction follows the same sequence:

1. Load and preprocess the image
2. Generate ResNet embedding
3. Apply the same PCA projection
4. Merge with corresponding tabular features
5. Apply the trained scaling/preprocessing path
6. Get class probabilities from RF + SVM + ANN
7. Average probabilities through soft voting
8. Return final landslide class prediction

## Model Artifact

The trained V1 ensemble model is serialized to:

- `models/ensemble_model.pkl`
- Hugging Face (DMv2 artifacts): https://huggingface.co/Nira00A/landslide-early-warning-ensemble

## Current Scope of V1

- V1 is designed as a multimodal baseline pipeline with classical ML ensemble fusion.
- Performance reporting currently includes accuracy, classification report, and confusion matrix from the evaluation split.

---

## DMv2: Latest Features

DMv2 extends the baseline pipeline with a deployment-focused inference workflow and updated training logic.

- **Hugging Face model delivery**: model artifacts are exported and loaded via `ensemble_artifactsV2.joblib` from `Nira00A/landslide-early-warning-ensemble`.
- **Sentinel-1 SAR imagery support**: image acquisition switched to Sentinel-1 GRD with robust fallback search windows and GCP-based pixel mapping.
- **Lower-dimensional image embeddings**: PCA reduced from 256 to 16 components for leaner inference.
- **Spatially aware split strategy**: tabular training pipeline now supports KMeans-based spatial grouping with `GroupShuffleSplit`.
- **Calibrated soft-voting ensemble**: RF, SVM, and ANN are calibrated (`CalibratedClassifierCV`) before voting.
- **Threshold-based decisioning**: `FixedThresholdClassifier` with threshold `0.40` is used to improve class decision behavior.
- **Live terrain/weather feature extraction**: Open-Meteo APIs provide elevation/slope/TRI and rainfall/soil moisture, with safe fallbacks if APIs fail.

## What Improved in the Latest Added Code (DMv2 vs V1)

- **Reduced false negatives focus** through calibrated probabilities, threshold tuning, and weighted soft voting (`[2,1,1]` for RF/SVM/ANN).
- **Better generalization and less overfitting risk** from ANN regularization updates (`alpha`, early stopping, validation fraction) and RF complexity control (`max_depth=8`, `min_samples_leaf=4`).
- **More reliable remote inference** by loading full artifacts bundle (model + PCA + scaler + threshold) directly from Hugging Face.
- **More robust geospatial ingestion** by replacing brittle local dependencies with API-based terrain/topology features and fallback defaults.