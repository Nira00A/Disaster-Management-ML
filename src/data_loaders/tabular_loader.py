from pandas import DataFrame
from pathlib import Path
import pandas as pd
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing._encoders import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from typing_extensions import Tuple, Optional 

## Data Directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "final_tabular_dataset.csv"))

## Loading the dataset of the Landslides Containing feilds like - id , latitude , longitude , geometry .....
def load_landlisde_tabular_data(
    data_dir: str = DATA_DIR):

    # Load dataset
    df = pd.read_csv(data_dir)

    print(df.shape)
    return df

## Train test split
def splitting_data(df: DataFrame, target_variable: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
    

## Scaling the data
def scale_data(Xtrain , Xtest , columns: list[str]):
    scale = StandardScaler()
    Xtrain_scaled = Xtrain.copy()
    Xtrain_scaled[columns] = scale.fit_transform(Xtrain[columns])

    Xtest_scaled = Xtest.copy()
    Xtest_scaled[columns] = scale.transform(Xtest[columns])
    return Xtrain_scaled, Xtest_scaled

if __name__ == "__main__":
    df = load_landlisde_tabular_data()
    print(df.head())
    '''Xtrain, Xtest, ytrain, ytest = splitting_data(df, "label")
    Xtrain_scaled, Xtest_scaled = scale_data(Xtrain, Xtest, ["soil_moisture_0_7cm_avg", "soil_moisture_7_28cm_avg", "daily_rainfall_mm", "cumulative_rainfall_3d_mm", "cumulative_rainfall_7d_mm", "elev_1", "slope_1", "rough_1", "tri_1"])
    print(Xtrain_scaled.head())
    print(Xtest_scaled.head())'''