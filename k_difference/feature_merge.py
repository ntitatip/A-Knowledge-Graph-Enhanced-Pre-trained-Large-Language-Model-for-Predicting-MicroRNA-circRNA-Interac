import os
import pandas as pd
import pickle

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep=",", header=None)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise pickle.PickleError("loading pickle error: {}".format(str(e)))


def merge_features(idx_path, feature_path):
    try:
        idx_name = os.path.basename(idx_path).split("_")[:2]
        feature_name = os.path.basename(feature_path).split("_")[:2]
        if idx_name != feature_name:
            return None  

    except Exception as e:
        print(f"match path name error: {e}")
        return None

    idx_df = read_csv(idx_path)
    feature_df = load_pickle(feature_path)
    tensor_data = [tensor.numpy() for tensor in feature_df]
    feature_df = pd.DataFrame(tensor_data)

    try:
        len(idx_df) == len(feature_df)
    except Exception as e:
        print(f"match idx length error: {e}")
        return None
    
    
    merged_df = pd.concat([idx_df.iloc[:, [0]], feature_df], axis=1)
    print(merged_df.tail())
    results_name = os.path.basename(feature_path)
    results_path = os.path.join(merge_feature_direcotory, 'merged_' + results_name)


    try:
        with open(results_path, 'wb') as file:
            pickle.dump(merged_df, file)
        print("File written successfully")
    except PermissionError as e:
        print(f"Permission denied: {e}")
        print("Please ensure the file is not in use and you have sufficient permissions.")


    return merged_df

pyfile_path = os.path.dirname(os.path.realpath(__file__))
raw_data_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw_data')
feature_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'feature')
merge_feature_direcotory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'merge_feature')


directories_data = [d for d in os.listdir(raw_data_directory) if os.path.isdir(os.path.join(raw_data_directory, d))]

def merge_feature_idx(idx_directory):
    directories_feature = [d for d in os.listdir(feature_directory) if os.path.isdir(os.path.join(feature_directory, d))]
    for directory in directories_feature:
        dir_path = os.path.join(feature_directory, directory)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                feature_df = merge_features(idx_directory, file_path)
    return feature_df



for directory in directories_data:
    dir_path = os.path.join(raw_data_directory, directory)
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            merge_feature_idx(file_path)



