import pickle
import numpy as np
import pandas as pd
import random
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import logging
import lightgbm as lgb

def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def read_csv(file):
    try:
        df = pd.read_csv(file, index_col=False, header = None)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def merge_features(tensor, row):
    # Find the features for the first name
    features_1 = tensor.loc[row[0]].values
    # Find the features for the second name
    features_2 = tensor.loc[row[1]].values
    # Concatenate the features
    features = np.concatenate([features_1, features_2])
    # Return the original columns and the features
    return pd.Series([row[0], row[1], row['label']] + list(features))

def cross_validate_model(df, classifier):

    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    fold_scores = {
        'ACC': [],
        'F1': [],
        'MCC': [],
        'AUROC': [],
        'AUPRC': [],
    }

    kde_plot = {
        'positive': [],
        'negative': [],
    }

    tprs = []
    fprs = []
    precisions = []
    recalls = []

    for fold_index, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        positive_proba = y_pred_proba[y_test == 1]

        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        y_pred_proba_negative = classifier.predict_proba(X_test)[:, 0]

        if fold_index == 1:
            kde_plot['positive'] = y_pred_proba[y_test == 1]
            kde_plot['negative'] = y_pred_proba[y_test == 0]


        fold_scores['ACC'].append(round(accuracy_score(y_test, y_pred),4))
        fold_scores['F1'].append(round(f1_score(y_test, y_pred),4))
        fold_scores['MCC'].append(round(matthews_corrcoef(y_test, y_pred),4))
        fold_scores['AUROC'].append(round(roc_auc_score(y_test, y_pred_proba),4))

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        fold_scores['AUPRC'].append(round(auc(recall, precision),4))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        tprs.append(tpr)
        fprs.append(fpr)

        # 计算Precision和Recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        precisions.append(precision)
        recalls.append(recall)

    return fold_scores


def training_crossvalidation_data(pair, tensor):
    tensor.columns = [str(i) for i in range(len(tensor.columns))]
    tensor.set_index(tensor.columns[0], inplace=True)

    nodes_attr1 = pair[0].unique()
    nodes_attr2 = pair[1].unique()

    # Get all positive pairs
    positive_pairs = set(tuple(x) for x in pair.values)

    # Initialize a list to store the negative samples
    negative_samples = []

    # Generate negative samples
    while len(negative_samples) < len(pair):
        # Randomly select a node from each attribute
        node1 = random.choice(nodes_attr1)
        node2 = random.choice(nodes_attr2)

        # Check if the pair is a positive sample
        if (node1, node2) not in positive_pairs:
            # If not, add it to the negative samples
            negative_samples.append((node1, node2))

    # Convert the negative samples to a DataFrame
    negative_df = pd.DataFrame(negative_samples, columns=pair.columns)
    negative_df['label'] = 0
    pair['label'] = 1

    train_df = pd.concat([pair, negative_df], ignore_index=True)

    features_df = train_df.apply(lambda row: merge_features(tensor, row), axis=1)

    return features_df

def training_test_data(df, clf):
    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    results = {}
    results['acc'] = round(accuracy_score(y_test, y_pred),4)
    results['f1'] = round(f1_score(y_test, y_pred),4)
    results['mcc'] =round(matthews_corrcoef(y_test, y_pred),4)
    results['auroc'] = round(roc_auc_score(y_test, y_score),4)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    results['auprc'] = round(auc(recall, precision),4)

    fpr, tpr, _ = roc_curve(y_test, y_score)

    # 计算PR曲线的值
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    curve_array = {'fpr':fpr ,'tpr':tpr,'precison': precision, 'recall':recall}

    # 找到原本标签为正的样本

    return curve_array, results
    





MAX_average_rounded = -1  # 初始化最高的mean_auc值为-1
best_i, best_j = None, None  # 初始化最高mean_auc值对应的i和j
lst = [3, 4, 5, 6]



csv_file9589 = 'C://backup//2024//BERT-DGI//graph_feature//9589_pair.csv'
csv_file9905 = 'C://backup//2024//BERT-DGI//graph_feature//9905_pair.csv'




clf_9589 = lgb.LGBMClassifier(n_estimators=69, max_depth=3, learning_rate=0.0403, verbosity=-1)
clf_9905 = lgb.LGBMClassifier(n_estimators=89, max_depth=2, learning_rate=0.0403, verbosity=-1)

#################################################bubbleplot############################################
# lst = [3, 4, 5, 6]
# new_lst = [(i, j) for i in lst for j in lst]



# results9905 = {}
# for tup in new_lst:
#     filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_{tup[0]}+miRNA_{tup[1]}_9905.pkl'
#     tensor9905 = load_pickle_file(filepath9905)
#     df9905 = read_csv(csv_file9905)
#     features_df9905 = training_crossvalidation_data(df9905,tensor9905)
#     roc_aupr, r = training_test_data(features_df9905, clf_9905)
#     results9905[tup] = r


# results9589 = {}
# for tup in new_lst:
#     filepath9589 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_{tup[0]}+miRNA_{tup[1]}_9589.pkl'
#     tensor9589 = load_pickle_file(filepath9589)
#     df9589 = read_csv(csv_file9589)
#     features_df9589 = training_crossvalidation_data(df9589,tensor9589)
#     roc_aupr, r = training_test_data(features_df9589, clf_9589)
#     results9589[tup] = r




# # 找到对应的键


# with open(f'C://backup//2024//BERT-DGI//results/bubble9589.pkl', 'wb') as f:
#     pickle.dump(results9589, f)

# with open(f'C://backup//2024//BERT-DGI//results/bubble9905.pkl', 'wb') as f:
#     pickle.dump(results9905, f)


##################################boxplot9905#############################################

# filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9905.pkl'
# df9905 = read_csv(csv_file9905)
# tensor9905= load_pickle_file(filepath9905)
# features_df9905 = training_crossvalidation_data(df9905,tensor9905)
# ours_boxplot9905 = cross_validate_model(features_df9905, clf_9905)


# with open(f'C://backup//2024//BERT-DGI//results/ours_boxplot9905.pkl', 'wb') as f:
#     pickle.dump(ours_boxplot9905, f)

##################################boxplot9589#############################################

# filepath9589 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9589.pkl'
# df9589 = read_csv(csv_file9589)
# tensor9589= load_pickle_file(filepath9589)
# features_df9589 = training_crossvalidation_data(df9589,tensor9589)
# ours_boxplot9589 = cross_validate_model(features_df9589, clf_9589)
# logging.info(f'ours_boxplot9589: {ours_boxplot9589}')


# with open(f'C://backup//2024//BERT-DGI//results/ours_boxplot9589.pkl', 'wb') as f:
#     pickle.dump(ours_boxplot9589, f)

##################################curveplot9589#############################################
filepath9589 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9589.pkl'
df9589 = read_csv(csv_file9589)
tensor9589= load_pickle_file(filepath9589)
features_df9589 = training_crossvalidation_data(df9589,tensor9589)
ours_curveplot9589, rounded_auc9589 = training_test_data(features_df9589, clf_9589)
print(f'ours_curveplot9589: {rounded_auc9589}')

# with open(f'C://backup//2024//BERT-DGI//results/additional9589.pkl', 'wb') as f:
#     pickle.dump(ours_curveplot9589, f)

##################################curveplot9905#############################################
filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9905.pkl'
df9905 = read_csv(csv_file9905)
tensor9905= load_pickle_file(filepath9905)
features_df9905 = training_crossvalidation_data(df9905,tensor9905)
ours_curveplot9905, rounded_auc9905 = training_test_data(features_df9905, clf_9905)
print(f'ours_curveplot9905: {rounded_auc9905}')

# with open(f'C://backup//2024//BERT-DGI//results/ours_curves9905.pkl', 'wb') as f:
#     pickle.dump(ours_curveplot9905, f)

##################################barplot9589#############################################

# filepath9589 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9589.pkl'
# df9589 = read_csv(csv_file9589)
# tensor9589= load_pickle_file(filepath9589)
# features_df9589 = training_crossvalidation_data(df9589,tensor9589)

# Classifiers = {
#     'Without Sequence Feature': lgb.LGBMClassifier(n_estimators=71, max_depth=1, learning_rate=0.0403, verbosity=-1),
#     'Without Network Feature': lgb.LGBMClassifier(n_estimators=77, max_depth=1, learning_rate=0.0403, verbosity=-1),
#     'All Features Combination': lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.0403, verbosity=-1),
# }

# barplot9589 = {}
# for key, value in Classifiers.items():
#     logging.info(f'processing {key}')
#     barplot9589[key] = cross_validate_model(features_df9589, value)

# with open(f'C://backup//2024//BERT-DGI//results/barplot9589.pkl', 'wb') as f:
#     pickle.dump(barplot9589, f)

# print(f'barplot9589: {barplot9589}')

##################################barplot9905#############################################

# filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9905.pkl'
# df9905 = read_csv(csv_file9905)
# tensor9905= load_pickle_file(filepath9905)
# features_df9905 = training_crossvalidation_data(df9905,tensor9905)

# Classifiers = {
#     'Without Sequence Feature': lgb.LGBMClassifier(n_estimators=65, max_depth=1, learning_rate=0.0403, verbosity=-1),
#     'Without Network Feature': lgb.LGBMClassifier(n_estimators=69, max_depth=1, learning_rate=0.0403, verbosity=-1),
#     'All Features Combination':lgb.LGBMClassifier(n_estimators=99, max_depth=2, learning_rate=0.0403, verbosity=-1)
# }

# barplot9905 = {}
# for key, value in Classifiers.items():
#     logging.info(f'processing {key}')
#     barplot9905[key] = cross_validate_model(features_df9905, value)

# with open(f'C://backup//2024//BERT-DGI//results/barplot9905.pkl', 'wb') as f:
#     pickle.dump(barplot9905, f)

# print(f'barplot9905: {barplot9905}')

