import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
import pickle

def get_valid_filename():
    # Function to get a valid filename
    while True:
        filename = input("Please enter the filename:\n")
        if os.path.isfile(filename):
            return filename
        else:
            print("Invalid filename. Please try again.")

def add_features(X):
    # Add four more features to the dataframe
    X["SqFtPerRoom"] = X["GrLivArea"] / (X["TotRmsAbvGrd"] + X["FullBath"] + X["HalfBath"] + X["KitchenAbvGr"])
    X['Total_Home_Quality'] = X['OverallQual'] + X['OverallCond']
    X['Total_Bathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath']) + X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))
    X["HighQualSF"] = X["1stFlrSF"] + X["2ndFlrSF"]

    # Ensure Proper Data Types
    X['MSSubClass'] = X['MSSubClass'].astype(str)
    X['YrSold'] = X['YrSold'].astype(str)
    X['MoSold'] = X['MoSold'].astype(str)

    return X

def preprocessor(X):
    # Add features
    X = add_features(X)

    # Define ordinal features
    feat_ordinal_dict = {
        # considers "missing" as "neutral"
        "BsmtCond": ['missing', 'Po', 'Fa', 'TA', 'Gd'],
        "BsmtExposure": ['missing', 'No', 'Mn', 'Av', 'Gd'],
        "BsmtFinType1": ['missing', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        "BsmtFinType2": ['missing', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        "BsmtQual": ['missing', 'Fa', 'TA', 'Gd', 'Ex'],
        "Electrical": ['missing', 'Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
        "ExterCond": ['missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "ExterQual": ['missing', 'Fa', 'TA', 'Gd', 'Ex'],
        "Fence": ['missing', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
        "FireplaceQu": ['missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "Functional": ['missing', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
        "GarageCond": ['missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "GarageFinish": ['missing', 'Unf', 'RFn', 'Fin'],
        "GarageQual": ['missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "HeatingQC": ['missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "KitchenQual": ['missing', 'Fa', 'TA', 'Gd', 'Ex'],
        "LandContour": ['missing', 'Low', 'Bnk', 'HLS', 'Lvl'],
        "LandSlope": ['missing', 'Sev', 'Mod', 'Gtl'],
        "LotShape": ['missing', 'IR3', 'IR2', 'IR1', 'Reg'],
        "PavedDrive": ['missing', 'N', 'P', 'Y'],
        "PoolQC": ['missing', 'Fa', 'Gd', 'Ex'],
    }

    feat_ordinal = sorted(feat_ordinal_dict.keys()) # sort alphabetically
    feat_ordinal_values_sorted = [feat_ordinal_dict[i] for i in feat_ordinal]

    # Define feat_numerical features
    feat_numerical = X.select_dtypes(include=["int64", "float64"]).columns

    # Define ordinal features
    feat_nominal = sorted(list(set(X.columns) - set(feat_numerical) - set(feat_ordinal)))

    encoder_ordinal = OrdinalEncoder(
        categories=feat_ordinal_values_sorted,
        dtype= np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1 # Considers unknown values as worse than "missing"
    )

    preproc_ordinal = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        encoder_ordinal,
        MinMaxScaler()
    )

    preproc_numerical = make_pipeline(
        KNNImputer(),
        MinMaxScaler()
    )

    preproc_nominal = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    preproc_transformer = make_column_transformer(
        (preproc_numerical, feat_numerical),
        (preproc_ordinal, feat_ordinal),
        (preproc_nominal, feat_nominal),
        remainder="drop")

    preproc_selector = SelectPercentile(
        mutual_info_regression,
        percentile=75, # keep only xx% of all features )
    )
    preprocessor = make_pipeline(
        preproc_transformer,
        preproc_selector
    )

    return preprocessor

def get_train_data():
    # Read the X_train and y_train data from separate files
    train = pd.read_csv("data/train.csv")
    data = train.drop(columns=['Id'])
    y = data.SalePrice
    y = np.log1p(y)
    X = data.drop(columns=['SalePrice'])
    return X, y

def model():
    # Load the trained_models dictionary from the file
    with open('trained_models_full.pkl', 'rb') as file:
        trained_models = pickle.load(file)
    return trained_models

def model_predict(trained_models, preproc_Xtest, test_id):
    # Ensemble the models for predictions
    final_predictions = (
        0.2 * np.exp(trained_models['GradientBoostingRegressor'].predict(preproc_Xtest)) +
        0.1 * np.exp(trained_models['LGBMRegressor'].predict(preproc_Xtest)) +
        0.4 * np.exp(trained_models['BayesianRidge'].predict(preproc_Xtest)) +
        0.1 * np.exp(trained_models['Ridge'].predict(preproc_Xtest)) +
        0.1 * np.exp(trained_models['ExtraTreesRegressor'].predict(preproc_Xtest)) +
        0.1 * np.exp(trained_models['RandomForestRegressor'].predict(preproc_Xtest))
    )

    # Create the prediction
    prediction = pd.DataFrame(test_id, columns=['Id'])
    prediction['SalePrice'] = pd.DataFrame(final_predictions)

    # Save the submission to a CSV file
    prediction.to_csv('submission/Prediction.csv', index=False, header=True)

    return prediction

# Main code
if __name__ == "__main__":
    # Get the train data
    X, y = get_train_data()

    # Preprocess the train data
    preprocessor = preprocessor(X)
    preproc_Xtrain = preprocessor.fit_transform(X,y)
    
    # Get a valid filename from the user
    filename = get_valid_filename()

    # Get the test CSV file as a DataFrame and drop the Id column
    test = pd.read_csv(filename)
    test_id = test.Id
    test = test.drop(columns=['Id'])

    # Add features and preprocess the test dataset
    preproc_Xtest = add_features(test)
    preproc_Xtest = preprocessor.transform(test)

    # Train the models
    trained_models = model()
    
    # Use the model to make prediction and save it as a csv file
    model_predict(trained_models, preproc_Xtest, test_id)
