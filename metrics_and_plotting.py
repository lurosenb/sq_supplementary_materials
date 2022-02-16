from acs_helper import ACSData

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from diffprivlib.models import LogisticRegression as DPLR
from sklearn.neural_network import MLPClassifier

import sklearn.metrics as skm
from fairlearn.metrics import MetricFrame
import fairlearn.metrics as fm
import xgboost as xgb 
from sklearn.metrics import f1_score 
from sklearn.utils import resample
import sdmetrics
import seaborn as sns

import pickle

dict_all_types_metrics = {
    "demographic_parity_difference":(0,(0,0.3)),
    "demographic_parity_ratio":(0,(0.8,1.0)),
    "equalized_odds_difference":(0,(0,0.4)),
    "equalized_odds_ratio":(0,(0.5,1.0)),
    "accuracy":(1,(0.5,0.9)),
    "precision":(1,(0.4,0.9)),
    "recall_score":(1,(0.6,1.0)),
    "f1_score":(1,(0.5,1.0)),
    "false_positive_rate":(1,(0,1.0)),
    "false_negative_rate":(1,(0,0.6)),
    "selection_rate":(1,(0,1.0)),
    "mean_prediction":(1,(0.4,1.0)),
    'CSTest':(2,(0,1.0)),
    'KSTestExtended':(2,(0,1.0)),
    'LogisticDetection':(2,(0,1.0)),
    'SVCDetection':(2,(0,1.0)),
    'BNLikelihood':(2,(0,1.0)),
    "model_score":(0,(0.5,1.0)),
}

MODELS = [('xgboost', None), ('MLP',MLPClassifier), ('RandomForest',RandomForestClassifier)]
REAL_FAKE = {0:"Real", 1:"Fake"}
REAL_FAKE_REVERSE = {"Real":0, "Fake":1}

def process_list_results(list_results, method="SuperQUAIL"):
    metrics = {}
    for sdm, real_fake_dict, real_fake_dict_filtered, real_data_fairness, fake_data_fairness in list_results:
        # SDMetrics Parse
        sdm_score_dict = dict(zip(sdm.metric, sdm.normalized_score))
        for key in sdm_score_dict.keys():
            if key not in metrics:
                metrics[key] = []
        for k, v in sdm_score_dict.items():
            metrics[k].append(v)

        # Modeling Metrics Parse
        modeling_keys = ['real_model_score','fake_model_score',"real_f1_score_filtered","fake_f1_score_filtered"]
        for key in modeling_keys:
            if key not in metrics:
                metrics[key] = []

        metrics["real_model_score"].append(real_fake_dict[REAL_FAKE_REVERSE["Real"]][3])
        metrics["fake_model_score"].append(real_fake_dict[REAL_FAKE_REVERSE["Fake"]][3])
        
        if real_fake_dict_filtered is not None:
            metrics["real_f1_score_filtered"].append(real_fake_dict_filtered[REAL_FAKE_REVERSE["Real"]][3])
            metrics["fake_f1_score_filtered"].append(real_fake_dict_filtered[REAL_FAKE_REVERSE["Fake"]][3])

        # Fairlearn
        if real_data_fairness is not None:
            fairlearn_dict_real = real_data_fairness[0].by_group.to_dict()
            for key in list(fairlearn_dict_real.keys()) + list(real_data_fairness[1].keys()):
                k = "real_"+key
                if k not in metrics:
                    metrics[k] = []
            for k, v in fairlearn_dict_real.items():
                k = "real_"+k
                metrics[k].append(v)
            for k, v in real_data_fairness[1].items():
                k = "real_"+k
                metrics[k].append(v)

            fairlearn_dict_fake = fake_data_fairness[0].by_group.to_dict()
            for key in list(fairlearn_dict_fake.keys()) + list(fake_data_fairness[1].keys()):
                k = "fake_"+key
                if k not in metrics:
                    metrics[k] = []
            for k, v in fairlearn_dict_fake.items():
                k = "fake_"+k
                metrics[k].append(v)
            for k, v in fake_data_fairness[1].items():
                k = "fake_"+k
                metrics[k].append(v)

    metrics["method"] = [method for i in range(0,len(list_results))]
    metrics = {k:v for k,v in metrics.items() if v}
    return metrics
    
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
        
    #Z Display new class counts
    print(df_resampled[target_col].value_counts())
    print(len(df_resampled))

    return df_resampled

def boost(x_train, y_train, x_test_real, y_test_real, prob_thresh=0.5, print_res=False):
    y_test = np.array(y_test_real).astype(int)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test_real)
    parameters = {'max_depth':14, 
                'n_estimators':1000,
                'eta':1, 
                'silent':1,
                'objective':'binary:logistic',
                'early_stopping_rounds':10,
                'eval_metric':'auc',
                'learning_rate':.01}
    num_round = 500
    xg = xgb.train(parameters,dtrain,num_round) 
    ypred = xg.predict(dtest) 

    for i in range(0, len(ypred)): 
        if ypred[i] >= prob_thresh:
            ypred[i] = 1 
        else: 
            ypred[i] = 0  

    if print_res:
        f1 = f1_score(y_test, ypred)
        print(f1)
        print(classification_report(y_test, ypred))

    return y_test, ypred, xg

def train_models(real, fake, models, verbose=False, test_size=0.2, random_state=42):
    real_fake = []
    fake = fake[real.columns]
    X = real.iloc[:, :-1]
    y = real.iloc[:, -1]
    # We want to save real data test set, as that is what we
    # evaluate on
    _, x_test_real, _, y_test_real = train_test_split(X, y, test_size=test_size, random_state=random_state)
    real_fake.append(train_test_split(X, y, test_size=test_size, random_state=random_state))

    X_fake = fake.iloc[:, :-1]
    y_fake = fake.iloc[:, -1]
    real_fake.append(train_test_split(X_fake, y_fake, test_size=test_size, random_state=random_state))
    
    real_fake_dict = {}
    for i, (x_train, _, y_train, _) in enumerate(real_fake):
        best_model = None
        best_model_score = 0.0
        best_ys = (None, None)
        for m, model in models:
            if m == 'xgboost':
                _, y_pred, trained_model = boost(x_train, y_train, x_test_real, y_test_real)
            else:
                if m == 'MLP':
                    trained_model = model(max_iter=1000)
                    trained_model.fit(x_train, y_train)
                else:
                    trained_model = model()
                    trained_model.fit(x_train, y_train)

                #Test the model
                y_pred = trained_model.predict(x_test_real)

            f1 = f1_score(y_test_real, y_pred)
            print(m)
            print(f1)
            print()
            
            if f1 > best_model_score:
                best_model = trained_model
                best_ys = (y_test_real, y_pred)
                best_model_score = f1
    
        print("Best model score " + REAL_FAKE[i] + ": " + str(best_model_score))
        real_fake_dict[i] = (best_ys[0], best_ys[1], best_model, best_model_score, x_test_real)

    return real_fake_dict
    
def calculate_fairness_metrics(y_test, 
                               ypred, 
                               group_membership_data, 
                               plot=False, 
                               print_name="Calculate Fairness Metrics"):
    dmd = fm.demographic_parity_difference(y_test, 
                                 ypred, 
                                 sensitive_features=group_membership_data)
    dpr = fm.demographic_parity_ratio(y_test, 
                                    ypred, 
                                    sensitive_features=group_membership_data)
    eod = fm.equalized_odds_difference(y_test, 
                                    ypred, 
                                    sensitive_features=group_membership_data)
    eor = fm.equalized_odds_ratio(y_test, 
                                    ypred, 
                                    sensitive_features=group_membership_data)
    
    CM = skm.confusion_matrix(y_test, ypred)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    metric_dict = {
        "demographic_parity_difference": dmd,
        "demographic_parity_ratio": dpr,
        "equalized_odds_difference": eod,
        "equalized_odds_ratio": eor,
        "overall_accuracy": skm.accuracy_score(y_test, ypred),
        "overall_precision": skm.precision_score(y_test, ypred),
        "overall_recall_score": skm.recall_score(y_test, ypred),
        "overall_f1_score": skm.f1_score(y_test, ypred),
        "overall_false_positive_rate": (float(FP)/float((FP+TN))),
        "overall_false_negative_rate": (float(FN)/float((FN+TP))),
        "overall_selection_rate": 0,
        "overall_count": 0,
        "overall_mean_prediction": 0,
    }

    metrics = {
        "accuracy": skm.accuracy_score,
        "precision": skm.precision_score,
        "recall_score": skm.recall_score,
        "f1_score": skm.f1_score,
        "false_positive_rate": fm.false_positive_rate,
        "false_negative_rate": fm.false_negative_rate,
        "selection_rate": fm.selection_rate,
        "count": fm.count,
        "mean_prediction": fm.mean_prediction,
    }

    metric_frame = MetricFrame(
        metrics=metrics, 
        y_true=y_test, 
        y_pred=ypred, 
        sensitive_features=group_membership_data
    )

    if plot:
        metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title=print_name,
        )
    return metric_frame, metric_dict

def calculate_all_metrics(real_data, 
                          fake_data, 
                          scenario="ACSEmployment", 
                          sample=False,
                          sample_frac=0.05, 
                          protected_class='RAC1P',
                          privileged_unpriveleged=[1.0,2.0],
                          plot=False):
    # Run all the compatible metrics and get a report
    real_data_with_id = real_data.reset_index()
    real_data_with_id = real_data_with_id.rename(columns = {'index':'id'})
    fake_data_with_id = fake_data.reset_index()
    fake_data_with_id = fake_data_with_id.rename(columns = {'index':'id'})
    if sample:
        real_data_with_id = real_data_with_id.sample(frac=sample_frac)
        fake_data_with_id = fake_data_with_id.sample(frac=sample_frac)

    metrics = sdmetrics.multi_table.MultiTableMetric.get_subclasses()
    mets = ['CSTest','KSTestExtended','LogisticDetection','SVCDetection','BNLikelihood']
    metrics = { key:value for (key,value) in metrics.items() if key in mets}
    
    acs = ACSData(states=['CA'])
    meta = acs.sdmetrics_metadata(real_data, scenario)
    sdm = sdmetrics.compute_metrics(metrics, {scenario: real_data_with_id}, {scenario: fake_data_with_id}, metadata=meta)
    
    real_fake_dict = train_models(real_data, fake_data, MODELS)

    real_data_fairness = None
    fake_data_fairness = None
    real_fake_dict_filtered = None

    if protected_class is not None:
        real_data_filtered = real_data[real_data[protected_class].isin(privileged_unpriveleged)]
        fake_data_filtered = fake_data[fake_data[protected_class].isin(privileged_unpriveleged)]
        real_fake_dict_filtered = train_models(real_data_filtered, fake_data_filtered, MODELS)

        real_data_fairness = calculate_fairness_metrics(real_fake_dict_filtered[0][0],
                                                        real_fake_dict_filtered[0][1],
                                                        real_fake_dict_filtered[0][4][protected_class],
                                                        print_name="Real Filtered",
                                                        plot=False)

        fake_data_fairness = calculate_fairness_metrics(real_fake_dict_filtered[1][0],
                                                        real_fake_dict_filtered[1][1],
                                                        real_fake_dict_filtered[1][4][protected_class],
                                                        print_name="Fake Filtered",
                                                        plot=False)
                                                    
    return sdm, real_fake_dict, real_fake_dict_filtered, real_data_fairness, fake_data_fairness

def generate_all_box_and_whisker(list_plot_box_whiskers, 
                                 metrics, 
                                 sensitive="race", 
                                 categories=['white','black'], 
                                 epsilon=3.0):
    print(list_plot_box_whiskers)
    for metric, (t, y_range) in metrics.items():
        exception_occurred = False
        try:
            new_data = []
            real = False
            for method, plot_box_whiskers in list_plot_box_whiskers:
                    real_plot = plot_box_whiskers
                    for index, row in real_plot.iterrows():
                        if t == 1:
                            if not real:
                                new_data.append((row['real_overall_'+str(metric)], "Overall", "Real"))
                                if sensitive is not None:
                                    new_data.append((row["real_"+str(metric)][1.0], categories[0], "Real"))
                                    new_data.append((row["real_"+str(metric)][2.0], categories[1], "Real"))
                                real=True
                            new_data.append((row['fake_overall_'+str(metric)], "Overall", method))
                            if sensitive is not None:
                                new_data.append((row["fake_"+str(metric)][1.0], categories[0], method))
                                new_data.append((row["fake_"+str(metric)][2.0], categories[1], method))
                        elif t == 0:
                            if not real:
                                new_data.append((row["real_"+str(metric)], categories[0], "Real"))
                                real=True
                            new_data.append((row["fake_"+str(metric)], categories[0], method))
                        else:
                            new_data.append((row[str(metric)], categories[0], method))
        except Exception as e:
            exception_occurred = True
            print("Metric probably doesn't exist")
            print(str(e))

        if not exception_occurred:
            for_whisker = pd.DataFrame(new_data, columns=[metric, sensitive, 'method'])
            plt.figure()
            if sensitive is not None:
                ax = sns.boxplot(x="method", 
                            y=metric, 
                            hue=sensitive,
                            data=for_whisker, 
                            palette="Set3")
                ax.set_title("ε="+str(epsilon)+", Comparing "+str(metric)+" Within Protected Groups")
                ax.set(ylim=y_range)
            else:
                ax = sns.boxplot(x="method", 
                            y=metric,
                            data=for_whisker)
                ax.set_title("ε="+str(epsilon)+", Comparing "+str(metric))
                ax.set(ylim=y_range)
            plt.savefig('figures/'+str(epsilon)+'_'+str(metric)+'.pdf',format='pdf')

def generate_list_results(pd_all_data, test_list, scenario):
    list_results = []
    for f1_avg, s, th, eps, sp1, sp2, f1s, crs, cols, list_samp in test_list:
        all_samps = pd.concat(list_samp)
        all_samps = resample_up_down(all_samps, upsample=True, target_col="ESR")
        
        list_results.append(calculate_all_metrics(
            pd_all_data,
            all_samps,
            sample=True, 
            scenario=scenario,
        ))
    return list_results