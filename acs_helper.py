from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSTravelTime, ACSMobility

import pandas as pd
import numpy as np

import json

class ACSData:
    """Wrapper for folktables to create the pandas dataframes we need."""
    ACS_Scenarios = {
        "ACSEmployment": ACSEmployment,
        "ACSIncome": ACSIncome,
        "ACSPublicCoverage": ACSPublicCoverage, 
        "ACSTravelTime": ACSTravelTime,
        "ACSMobility": ACSMobility
    }

    def __init__(self, states=None):
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        # self.acs_data = data_source.get_data(states=["CA"], download=True)
        self.acs_data = data_source.get_data(states=states, download=True)

    def return_acs_data_scenario(self, scenario="ACSEmployment", subsample=None, verbose=False):
        scenario = self.ACS_Scenarios[scenario]
        features, label, group = scenario.df_to_numpy(self.acs_data)
        if verbose:
            print(features, label, group)

        np_all_data = np.c_[features,label]

        if subsample is not None:
            np_all_data = np_all_data[np.random.choice(np_all_data.shape[0], subsample, replace=False)]

        pd_all_data = pd.DataFrame(np_all_data, columns = scenario._features + [scenario._target])
        pd_features = pd.DataFrame(features, columns = scenario._features)
        pd_target = pd.DataFrame(label, columns = [scenario._target])
        pd_group = pd.DataFrame(group, columns = [scenario._group])
        return pd_all_data, pd_features, pd_target, pd_group

    def return_simple_acs_data_scenario(self, scenario="ACSEmployment", subsample=None, verbose=False):
        """
        "Simple scenarios" are defined as just categorical (f_types=[0])
        """
        pd_all_data, pd_features, pd_target, pd_group = self.return_acs_data_scenario(
                                                                    scenario=scenario, 
                                                                    subsample=subsample, 
                                                                    verbose=verbose)
        allowed_features = self.get_metadata_features(f_types=[0])
        allowed_features = list(allowed_features.keys())

        # If a non-categorical column is target/group, make an exception
        if pd_target.columns[0] not in allowed_features:
            allowed_features.append(pd_target.columns[0])
        
        if pd_group.columns[0] not in allowed_features:
            allowed_features.append(pd_group.columns[0])
        
        remove_features = [f for f in pd_all_data.columns if f not in allowed_features]
        print(remove_features)
        pd_all_data = pd_all_data.drop(remove_features, axis=1)
        pd_features = pd_features.drop(remove_features, axis=1)

        return pd_all_data, pd_features, pd_target, pd_group
    
    def return_large_categorical_acs_data_scenario(self, scenario="ACSEmployment", subsample=None, verbose=False):
        """
        "Simple scenarios" are defined as just categorical (f_types=[0])
        """
        pd_all_data, pd_features, pd_target, pd_group = self.return_acs_data_scenario(
                                                                    scenario=scenario, 
                                                                    subsample=subsample, 
                                                                    verbose=verbose)
        allowed_features = self.get_metadata_features(f_types=[0,1])
        allowed_features = list(allowed_features.keys())

        # If a non-categorical column is target/group, make an exception
        if pd_target.columns[0] not in allowed_features:
            allowed_features.append(pd_target.columns[0])
        
        if pd_group.columns[0] not in allowed_features:
            allowed_features.append(pd_group.columns[0])
        
        remove_features = [f for f in pd_all_data.columns if f not in allowed_features]
        print(remove_features)
        pd_all_data = pd_all_data.drop(remove_features, axis=1)
        pd_features = pd_features.drop(remove_features, axis=1)

        return pd_all_data, pd_features, pd_target, pd_group

    def get_acs_names_features(self, verbose=False):
        acs_feature_dict = {}
        for n, scen in [("ACSEmployment", ACSEmployment), 
               ("ACSIncome", ACSIncome), 
               ("ACSPublicCoverage", ACSPublicCoverage), 
               ("ACSTravelTime", ACSTravelTime)]:
            acs_feature_dict[n] = scen._features
            if verbose:
                print(n)
                print(scen._features)
        return acs_feature_dict

    def sdmetrics_metadata(self, columns, name):
        meta = {
                "tables":{
                    name:{
                        "primary_key":"id",
                        "fields": {
                            'id':{
                                'type': 'id', 
                                'subtype': 'integer'
                            },
                        }
                    }
                }
            }
        fields = {
            "AGEP": {
                "type":"ordinal"
            },
            "ANC": {
                "type":"categorical"
            },
            "CIT": {
                "type":"categorical"
            },
            "COW": {
                "type":"categorical"
            },
            "DEAR": {
                "type":"categorical"
            },
            "DEYE": {
                "type":"categorical"
            },
            "DIS": {
                "type":"categorical"
            },
            "DREM": {
                "type":"categorical"
            },
            "ESP": {
                "type":"categorical"
            },
            "ESR": {
                "type":"categorical"
            },
            "FER": {
                "type":"categorical"
            },
            "JWTR": {
                "type":"categorical"
            },
            "MAR": {
                "type":"categorical"
            },
            "MIG": {
                "type":"categorical"
            },
            "MIL": {
                "type":"categorical"
            },
            "NATIVITY": {
                "type":"categorical"
            },
            "OCCP": {
                "type":"categorical"
            },
            "PINCP": {
                "type":"categorical"
            },
            "POBP": {
                "type":"categorical"
            },
            "POVPIP": {
                "type":"numerical"
            },
            "POWPUMA": {
                "type":"categorical"
            },
            "PUMA": {
                "type":"categorical"
            },
            "RAC1P": {
                "type":"categorical"
            },
            "GCL": {
                "type":"categorical"
            },
            "RELP": {
                "type":"categorical"
            },
            "SCHL": {
                "type":"categorical"
            },
            "SEX": {
                "type":"categorical"
            },
            "ST": {
                "type":"categorical"
            },
            "WKHP": {
                "type":"ordinal"
            },
            "PUBCOV": {
                "type":"categorical"
            },
            "JWMNP": {
                "type":"categorical"
            },
        }
        
        for c in columns:
            meta["tables"][name]["fields"][c] = fields[c]

        return meta
        
    def get_metadata_features(self, f_types=None):
        """
        f_type is list of feature types to return

        Note codes:
            Categorial: 0
            Large_catgorical: 1 (greater than 10 categories)
            Ordinal: 2
            Continuous: 3
        """
        feature_metadata = {
            "AGEP": 2,
            "ANC": 0,
            "CIT": 0,
            "COW": 0,
            "DEAR": 0,
            "DEYE": 0,
            "DIS": 0,
            "DREM": 0,
            "ESP": 0,
            "ESR": 0,
            "FER": 0,
            "JWTR": 1,
            "MAR": 0,
            "MIG": 0,
            "MIL": 0,
            "MIL": 0,
            "NATIVITY": 0,
            "OCCP": 2,
            "PINCP": 0,
            "POBP": 1,
            "POVPIP": 3,
            "POWPUMA": 2,
            "PUMA": 2,
            "RAC1P": 0,
            "RELP": 1,
            "SCHL": 1,
            "SEX": 0,
            "ST": 2,
            "WKHP": 2,
            "PUBCOV": 0,
            "JWMNP":0,
        }
        if f_types is None:
            return feature_metadata
        else:
            features_to_return = []
            for f_t in f_types:
                keys = [k for k, v in feature_metadata.items() if v == f_t]
                features_to_return = features_to_return + keys
            return dict((key, feature_metadata[key]) for key in features_to_return if key in feature_metadata)

# https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt
# For codes