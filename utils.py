import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.metrics import f1_score 
import sklearn.metrics as skm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from acs_helper import ACSData

def resample_up_down(dataframe, upsample=True, target_col="ESR", seed=0):
    # Separate majority and minority classes
    if len(dataframe[dataframe[target_col]==1]) > len(dataframe[dataframe[target_col]==0]):
        df_majority = dataframe[dataframe[target_col]==1]
        df_minority = dataframe[dataframe[target_col]==0]
    else:
        df_majority = dataframe[dataframe[target_col]==0]
        df_minority = dataframe[dataframe[target_col]==1]
    
    if upsample:
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                        replace=True,
                                        n_samples=len(df_majority),
                                        random_state=seed)
    
        # Combine majority class with upsampled minority class
        df_resampled = pd.concat([df_majority, df_minority_upsampled])
    else:
        # Downsample majority class
        df_majority_downsampled = resample(df_majority, 
                                        replace=False,
                                        n_samples=len(df_minority),
                                        random_state=seed) 
        
        # Combine minority class with downsampled majority class
        df_resampled = pd.concat([df_majority_downsampled, df_minority])
        
    # Display new class counts
    print(df_resampled[target_col].value_counts())
    print(len(df_resampled))

    return df_resampled

def prep_real(real, verbose=False):
    X = real.iloc[:, :-1]
    y = real.iloc[:, -1]
    _, x_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_test, y_test

def prep_real_with_model(real, model, verbose=False):
    X = real.iloc[:, :-1]
    y = real.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_real = model() #(max_iter=1000)
    model_real.fit(x_train, y_train)

    #Test the model
    predictions = model_real.predict(x_test)
    f1 = f1_score(y_test, predictions)
    if verbose:
        print()
        print('Trained on Real Data')
        print(classification_report(y_test, predictions))
        print('Accuracy real: ' + str(accuracy_score(y_test, predictions)))
        print(f1)
    return x_test, y_test, f1

def get_scenario(scenario):
    if "ACSEmployment":
        return get_employment()
    elif "ACSMobility":
        return get_mobility()
    elif "ACSPublicCoverage":
        return get_public()
    else:
        raise ValueError("Not supported: " + scenario)

def get_employment():
    acs = ACSData(states=['CA'])
    pd_all_data, pd_features, pd_target, pd_group = acs.return_simple_acs_data_scenario(scenario="ACSEmployment")
    return pd_all_data, pd_features, pd_target, pd_group

def get_mobility():
    acs = ACSData(states=["NM"])
    pd_all_data, pd_features, pd_target, pd_group = acs.return_simple_acs_data_scenario(scenario="ACSMobility")
    for col in pd_features.columns:
        codes = dict([(category, code) for code, category in enumerate(pd_all_data[col].unique())])
        pd_all_data[col] = pd_all_data[col].map(codes)
    pd_all_data = pd_all_data.drop("JWMNP", axis=1)
    pd_features = pd_features.drop("JWMNP", axis=1)
    pd_all_data = pd_all_data.drop("PINCP", axis=1)
    pd_features = pd_features.drop("PINCP", axis=1)
    pd_all_data = pd_all_data.drop("ESP", axis=1)
    pd_features = pd_features.drop("ESP", axis=1)

    return pd_all_data, pd_features, pd_target, pd_group

def get_public():
    acs = ACSData(states=["NM"])
    scenario = "ACSPublicCoverage"
    pd_all_data, pd_features, pd_target, pd_group = acs.return_simple_acs_data_scenario(scenario=scenario)
    pd_all_data = pd_all_data.drop("PINCP", axis=1)
    pd_features = pd_features.drop("PINCP", axis=1)
    for col in pd_features.columns:
        codes = dict([(category, code) for code, category in enumerate(pd_all_data[col].unique())])
        pd_all_data[col] = pd_all_data[col].map(codes)
    return pd_all_data, pd_features, pd_target, pd_group