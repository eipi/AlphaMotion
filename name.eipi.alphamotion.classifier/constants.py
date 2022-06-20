DATA_BASE_PATH = 'data'

FEATURE_COLUMNS = [
    'x_mean', 'y_mean', 'z_mean',
    'x_std', 'y_std', 'z_std',
    'num_x_peaks', 'num_y_peaks', 'num_z_peaks',
    'x_peak_prominence', 'y_peak_prominence', 'z_peak_prominence',
    'x_peak_width', 'y_peak_width', 'z_peak_width',
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
