DATA_BASE_PATH = 'data'

RAW_COLUMNS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'target']

SENSOR_TYPES = ['acc', 'gyro']
DIMENSIONS = ['x', 'y', 'z']
FEATURE_MEAN = 'mean'
FEATURE_STD = 'std'
FEATURE_NUM_PEAKS = 'num_peaks'
FEATURE_PEAK_PROMINENCE = 'peak_prominence'
FEATURE_PEAK_WIDTH = 'peak_width'
FEATURES = [FEATURE_MEAN, FEATURE_STD, FEATURE_NUM_PEAKS, FEATURE_PEAK_PROMINENCE, FEATURE_PEAK_WIDTH]
TARGET = 'target'


def get_feature_columns():
    feature_columns = []
    for sensor_type in SENSOR_TYPES:
        for dimension in DIMENSIONS:
            for feature in FEATURES:
                col_name = sensor_type + '_' + dimension + '_' + feature
                feature_columns.append(col_name)
    feature_columns.append(TARGET)
    return feature_columns


ACTIVITY_LABELS = {
    1: 'Walking',
    2: 'Walk Upstairs',
    3: 'Walk Downstairs',
    4: 'Sitting',
    5: 'Standing',
    6: 'Laying',
    7: 'Stand to sit',
    8: 'Sit to stand',
    9: 'Sit to lie',
    10: 'Lie to sit',
    11: 'Stand to lie',
    12: 'Lie to stand'
}
