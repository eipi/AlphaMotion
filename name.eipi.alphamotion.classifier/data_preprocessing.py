import traceback

import numpy as np

from utilities import feature_engineer
from constants import FEATURE_COLUMNS, OLEKS_ACTIVITY_LIST, DATA_BASE_PATH
from data_access import erenaktas_target_parser

import pandas as pd
import os

DATA_NUM = 2


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

def extractAndProcessRawDataFiles(sub_path):
    dataframe = pd.DataFrame(columns=FEATURE_COLUMNS)
    classification_index = erenaktas_target_parser("data/erenaktas/labels.txt")
    activity_files = os.listdir(os.path.join(DATA_BASE_PATH, sub_path))
    for file in activity_files:
        filename1 = file.split('acc_exp')
        filename2 = filename1[1]
        filename3 = filename2.split('_user')
        filename4 = filename3[0]
        exp_num = int(filename4)

        row_count = 0
        df = pd.read_csv(os.path.join(DATA_BASE_PATH, sub_path, file), sep=' ', header=None)
        df.columns = ["x", "y", "z"]
        # todo - review adding gyroscopic data
        num_samples = len(df.index)
        target_activity_group = classification_index[str(exp_num)]

        while row_count < num_samples:
            num_rows_to_parse = min(num_samples - row_count, 300)
            for i in range(1, num_rows_to_parse):
                vectors = []
                vectors.append(np.array(df.iloc(row_count + i - 1)))
            vector_df = pd.DataFrame(pd.Series(vectors))
            for sample_start in target_activity_group.keys():
                activity = target_activity_group[sample_start]
                row_count = row_count + num_samples

                dataframe = feature_engineer(
                    action=df.to_numpy(),
                    target=activity,
                    df=vector_df
                )
    #print(dataframe['target'].value_counts())
    dataframe['target'].value_counts().plot(kind='barh')
    dataframe.to_csv('data/temp/final_data.csv', index=False)

