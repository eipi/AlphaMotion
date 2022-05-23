from utilities import feature_engineer
from constants import FEATURE_COLUMNS, DATA_BASE_PATH
from data_access import erenaktas_target_parser
from data_manipulation import find_dominant_class_for_samples, Sample

import pandas as pd
import os

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

def extractAndProcessRawDataFiles(sub_path, slice_size, results_folder):
    dataframe = pd.DataFrame(columns=FEATURE_COLUMNS)
    classification_index = erenaktas_target_parser("data/erenaktas/labels.txt")
    activity_files = os.listdir(os.path.join(DATA_BASE_PATH, sub_path))
    total_number_of_frames_in_test = 0
    for file in activity_files:
        filename1 = file.split('acc_exp')
        filename2 = filename1[1]
        filename3 = filename2.split('_user')
        filename4 = filename3[0]
        exp_num = int(filename4)

        print("Processing experiment number " + filename4)

        row_count = 0
        df = pd.read_csv(os.path.join(DATA_BASE_PATH, sub_path, file), sep=' ', header=None)
        df.columns = ["x", "y", "z"]
        # todo - review adding gyroscopic data
        num_samples = len(df.index)

        target_activity_group = classification_index[str(exp_num)]

        sample_windows = []
        sample_vectors = []

        while row_count < num_samples:
            num_rows_to_parse = min(num_samples - row_count, slice_size)
            sample_vector = df[row_count:row_count + num_rows_to_parse].to_numpy()
            sample_vectors.append(sample_vector)
            sample_windows.append([row_count, row_count + num_rows_to_parse])
            row_count = row_count + num_rows_to_parse

        print('* Processed ' + str(len(sample_windows)) + ' frames')
        total_number_of_frames_in_test = total_number_of_frames_in_test + len(sample_windows)

        classification_map = find_dominant_class_for_samples(target_activity_group, sample_windows)
        for i in range(len(sample_windows)):
            sample_window = sample_windows[i]
            sample_vector = sample_vectors[i]
            sample_object = Sample(sample_window[0], sample_window[1])
            if sample_object in classification_map.keys():
                activity = classification_map.get(sample_object)
                if activity is not None:
                    dataframe = feature_engineer(
                        action=sample_vector,
                        target=activity,
                        df=dataframe
                    )
                else:
                    print('** Dropping 1 unclassified frame @ index ' + str(i))

    dataframe['target'].value_counts().plot(kind='barh')
    dataframe.to_csv(results_folder + '/final_data.csv', index=False)
    return total_number_of_frames_in_test

