import pandas as pd
import os
from constants import DATA_BASE_PATH


# example of data that AndroSensor return
pd.read_csv(os.path.join(DATA_BASE_PATH, 'running/running-1.csv')).head()


def preparator(df, action_name, path, rows_number=ROWS_NUMBER, step_size=STEP_SIZE):
    metrics_number = df.shape[0] // step_size - 14
    if metrics_number == 0:
        return 'Not enough rows...'
    index = 1
    borders = [0, rows_number]
    while index <= metrics_number:
        action_df = pd.DataFrame(columns=COLUMN_NAMES)
        action_df[COLUMN_NAMES[0]] = df[OLD_COLUMN_NAMES[0]][borders[0]:borders[1]]
        action_df[COLUMN_NAMES[1]] = df[OLD_COLUMN_NAMES[1]][borders[0]:borders[1]]
        action_df[COLUMN_NAMES[2]] = df[OLD_COLUMN_NAMES[2]][borders[0]:borders[1]]
        file_name = os.path.join(path, action_name + '-' + str(index) + '.csv')
        action_df.to_csv(file_name, index=False)

        # preparation for next step
        borders[0] = borders[0] + step_size
        borders[1] = borders[1] + step_size
        del action_df
        index += 1
    return 'OK'

