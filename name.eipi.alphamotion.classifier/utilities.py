import numpy as np
from scipy.signal import find_peaks


def mean_calculator(three_axis):
    """ Return mean of each vectors """
    three_axis = np.array(three_axis)
    vector_x = three_axis[:, 0]
    vector_y = three_axis[:, 1]
    vector_z = three_axis[:, 2]
    x_mean = np.mean(vector_x)
    y_mean = np.mean(vector_y)
    z_mean = np.mean(vector_z)
    return x_mean, y_mean, z_mean


def std_calculator(three_axis):
    """ Return standart deviation of each vectors """
    three_axis = np.array(three_axis)
    vector_x = three_axis[:, 0]
    vector_y = three_axis[:, 1]
    vector_z = three_axis[:, 2]
    x_std = np.std(vector_x)
    y_std = np.std(vector_y)
    z_std = np.std(vector_z)
    return x_std, y_std, z_std


def peaks_calculator(three_axis):
    """ Return number of peaks of each vectors """
    three_axis = np.array(three_axis)
    vector_x = three_axis[:, 0]
    vector_y = three_axis[:, 1]
    vector_z = three_axis[:, 2]
    x_peaks = len(find_peaks(vector_x)[0])
    y_peaks = len(find_peaks(vector_y)[0])
    z_peaks = len(find_peaks(vector_z)[0])
    return x_peaks, y_peaks, z_peaks


def feature_engineer(action, target, df):
    try:
        x_mean, y_mean, z_mean = mean_calculator(action)
        x_std, y_std, z_std = std_calculator(action)
        x_peaks, y_peaks, z_peaks = peaks_calculator(action)
    except:
        print(action.shape, target)

    dictionary = {
        'x_mean': x_mean,
        'y_mean': y_mean,
        'z_mean': z_mean,
        'x_std': x_std,
        'y_std': y_std,
        'z_std': z_std,
        'x_peaks': x_peaks,
        'y_peaks': y_peaks,
        'z_peaks': z_peaks,
        'target': target
    }
    df = df.append(
        dictionary,
        ignore_index=True
    )
    return df

