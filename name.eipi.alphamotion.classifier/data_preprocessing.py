from utilities import feature_engineer, pass_through
from constants import FEATURE_COLUMNS, DATA_BASE_PATH, RAW_COLUMNS
from data_access import erenaktas_target_parser
from data_manipulation import find_dominant_class_for_samples, Sample

import pandas as pd
import os

use_feature_extraction = False

def extractSingleFile(file_type, file_num):
    dataframe = pd.DataFrame(columns=FEATURE_COLUMNS[0:9])
    file = os.path.join(DATA_BASE_PATH, file_type, file_type + '-' + str(file_num) + '.csv')
    try:
        df = pd.read_csv(file)
        dataframe = feature_engineer(
            action=df.to_numpy(),
            target=file_type,
            df=dataframe
        )
    except:
        print('some error')
    return dataframe.drop('target', 1)

def process_raw_data(data_path, slice_size, results_folder):
    if use_feature_extraction:
        dataframe = pd.DataFrame(columns=FEATURE_COLUMNS)
    else:
        dataframe = pd.DataFrame(columns=RAW_COLUMNS)

    classification_index = erenaktas_target_parser("data/erenaktas/labels.txt")
    acc_files = os.listdir(os.path.join(DATA_BASE_PATH, data_path, 'acc'))
    gyro_files = os.listdir(os.path.join(DATA_BASE_PATH, data_path, 'gyro'))

    total_number_of_frames_in_test = 0
    print('Processing experiments ', end=" ")
    for file in acc_files:
        filename1 = file.split('acc_exp')
        filename2 = filename1[1]
        filename3 = filename2.split('_user')
        filename4 = filename3[0]
        exp_num = int(filename4)

        print(filename4, end = ", ")

        row_count = 0
        df = pd.read_csv(os.path.join(DATA_BASE_PATH, data_path, 'acc', file), sep=' ', header=None)
        df.columns = ["x", "y", "z"]

        num_samples = len(df.index)

        target_activity_group = classification_index[str(exp_num)]

        sample_windows = []
        acc_sample_vectors = []
        gyro_sample_vectors = []

        gyro_df = pd.read_csv(os.path.join(DATA_BASE_PATH, data_path, 'gyro', file.replace('acc', 'gyro')), sep=' ', header=None)
        gyro_df.columns = ["x", "y", "z"]

        while row_count < num_samples:
            num_rows_to_parse = min(num_samples - row_count, slice_size)
            acc_sample_vector = df[row_count:row_count + num_rows_to_parse].to_numpy()
            gyro_sample_vector = gyro_df[row_count:row_count + num_rows_to_parse].to_numpy()
            acc_sample_vectors.append(acc_sample_vector)
            gyro_sample_vectors.append(gyro_sample_vector)
            sample_windows.append([row_count, row_count + num_rows_to_parse])
            row_count = row_count + num_rows_to_parse

        #print('* Processed ' + str(len(sample_windows)) + ' frames')
        total_number_of_frames_in_test = total_number_of_frames_in_test + len(sample_windows)

        classification_map = find_dominant_class_for_samples(target_activity_group, sample_windows)
        for i in range(len(sample_windows)):
            sample_window = sample_windows[i]
            acc_sample_vector = acc_sample_vectors[i]
            gyro_sample_vector = gyro_sample_vectors[i]
            sample_object = Sample(sample_window[0], sample_window[1])
            if sample_object in classification_map.keys():
                activity = classification_map.get(sample_object)
                if activity is not None:
                    if use_feature_extraction:
                        dataframe = feature_engineer(
                            acc_data=acc_sample_vector,
                            gyro_data=gyro_sample_vector,
                            target=activity,
                            df=dataframe
                        )
                    else:
                        dataframe = feature_engineer(
                            acc_data=acc_sample_vector,
                            gyro_data=gyro_sample_vector,
                            target=activity,
                            df=dataframe
                        )
                else:
                    print('** Dropping 1 unclassified frame @ index ' + str(i))
    print(".")
    dataframe['target'].value_counts().plot(kind='barh')
    dataframe.to_csv(results_folder + '/final_data.csv', index=False)
    return total_number_of_frames_in_test, total_number_of_frames_in_test

