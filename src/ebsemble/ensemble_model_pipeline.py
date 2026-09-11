import pickle
from pathlib import Path
# pyrefly: ignore [missing-import]
from ..fusion_models import encoding, join_embeddings_with_tabular_data, preprocess_combined_data
# pyrefly: ignore [missing-import]
from ..data_loaders import load_landlisde_tabular_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "ensemble_model.pkl"

## Using the Resnet-50 (50 layer model) to derive the image's encodings
img_embeddings = encoding()

## Joining the image embeddings with the tabular data
combined_data = join_embeddings_with_tabular_data(img_embeddings, load_landlisde_tabular_data())

## Preprocessing the combined data
X_train, X_test, y_train, y_test = preprocess_combined_data(combined_data, target_variable="label")

'''print(X_train.info())
print(X_train.isnull().sum())
print(X_test.info())
print(y_train.info())
print(y_test.info())

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)'''

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

ensemble_model.fit(X_train, y_train)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(ensemble_model, f)

y_pred = ensemble_model.predict(X_test)

print(y_pred)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))





