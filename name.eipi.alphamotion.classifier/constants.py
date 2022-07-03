DATA_BASE_PATH = 'data'

RAW_COLUMNS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'target']

FEATURE_COLUMNS = [
    'acc_x_mean', 'acc_y_mean', 'acc_z_mean',
    'acc_x_std', 'acc_y_std', 'acc_z_std',
    'acc_num_x_peaks', 'acc_num_y_peaks', 'acc_num_z_peaks',
    'acc_x_peak_prominence', 'acc_y_peak_prominence', 'acc_z_peak_prominence',
    'acc_x_peak_width', 'acc_y_peak_width', 'acc_z_peak_width',
    'gyro_x_mean', 'gyro_y_mean', 'gyro_z_mean',
    'gyro_x_std', 'gyro_y_std', 'gyro_z_std',
    'gyro_num_x_peaks', 'gyro_num_y_peaks', 'gyro_num_z_peaks',
    'gyro_x_peak_prominence', 'gyro_y_peak_prominence', 'gyro_z_peak_prominence',
    'gyro_x_peak_width', 'gyro_y_peak_width', 'gyro_z_peak_width',
    'target'
]

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
