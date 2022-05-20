SAMPLE_FREQUENCY = 0.5

DATA_BASE_PATH = 'data'
DATA_IDLE = 'idle'
DATA_IDLE_NUM = 1039
DATA_RUNNING = 'running'
DATA_RUNNING_NUM = 3408
DATA_STAIRS = 'stairs'
DATA_STAIRS_NUM = 165
DATA_WALKING = 'walking'
DATA_WALKING_NUM = 1850


OLEKS_ACTIVITY_LIST = [DATA_IDLE, DATA_STAIRS, DATA_RUNNING, DATA_WALKING]

OLEKS_DATA_SUMMARY = {
    DATA_IDLE: DATA_IDLE_NUM,
    DATA_WALKING: DATA_WALKING_NUM,
    DATA_RUNNING: DATA_RUNNING_NUM,
    DATA_STAIRS: DATA_STAIRS_NUM
}


FEATURE_COLUMNS = [
    'x_mean', 'y_mean', 'z_mean',
    'x_std', 'y_std', 'z_std',
    'num_x_peaks', 'num_y_peaks', 'num_z_peaks',
    'x_peak_prominence', 'y_peak_prominence', 'z_peak_prominence',
    'x_peak_width', 'y_peak_width', 'z_peak_width',
    'target'
]


