from data_access import read_csv_file_to_numpy
from utilities import feature_engineer
from constants import FEATURE_COLUMNS, ACTIVITY_LIST, DATA_BASE_PATH
import utilities
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


def extractAndProcessAllDataFiles():
    dataframe = pd.DataFrame(columns=FEATURE_COLUMNS)
    for activity in ACTIVITY_LIST:
        activity_files = os.listdir(os.path.join(DATA_BASE_PATH, activity))
        for file in activity_files:
                try:
                    df = pd.read_csv(os.path.join(DATA_BASE_PATH, activity, file))
                    dataframe = feature_engineer(
                        action=df.to_numpy(),
                        target=activity,
                        df=dataframe
                    )
                except:
                    print('some error')
    # data = [x, y, z]
    # generate_plots(x, y, z, 3)
    print(dataframe['target'].value_counts())
    dataframe['target'].value_counts().plot(kind='barh')
    dataframe.to_csv('data/final_data.csv', index=False)

