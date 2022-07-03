import numpy as np
from numpy import average
from scipy.signal import find_peaks, peak_prominences, peak_widths


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

    x_peaks = find_peaks(vector_x)[0]
    y_peaks = find_peaks(vector_y)[0]
    z_peaks = find_peaks(vector_z)[0]

    num_x_peaks = len(x_peaks)
    num_y_peaks = len(y_peaks)
    num_z_peaks = len(z_peaks)

    x_peak_prominences = np.mean(peak_prominences(vector_x, x_peaks))
    y_peak_prominences = np.mean(peak_prominences(vector_y, y_peaks))
    z_peak_prominences = np.mean(peak_prominences(vector_z, z_peaks))

    x_peak_widths = np.mean(peak_widths(vector_x, x_peaks))
    y_peak_widths = np.mean(peak_widths(vector_y, y_peaks))
    z_peak_widths = np.mean(peak_widths(vector_z, z_peaks))

    return num_x_peaks, num_y_peaks, num_z_peaks, \
           x_peak_prominences, y_peak_prominences, z_peak_prominences, \
           x_peak_widths, y_peak_widths, z_peak_widths


# find_peaks       -- Find a subset of peaks inside a signal.
# find_peaks_cwt   -- Find peaks in a 1-D array with wavelet transformation.
# peak_prominences -- Calculate the prominence of each peak in a signal.
# peak_widths      -- Calculate the width of each peak in a signal.
def feature_engineer(acc_data, gyro_data, target, df):
    try:
        x_mean, y_mean, z_mean = mean_calculator(acc_data)
        x_std, y_std, z_std = std_calculator(acc_data)
        num_x_peaks, num_y_peaks, num_z_peaks, x_peak_prominence, y_peak_prominence, z_peak_prominence,\
            x_peak_width, y_peak_width, z_peak_width = peaks_calculator(acc_data)
        gyro_x_mean, gyro_y_mean, gyro_z_mean = mean_calculator(gyro_data)
        gyro_x_std, gyro_y_std, gyro_z_std = std_calculator(gyro_data)
        gyro_num_x_peaks, gyro_num_y_peaks, gyro_num_z_peaks, gyro_x_peak_prominence, gyro_y_peak_prominence, gyro_z_peak_prominence,\
            gyro_x_peak_width, gyro_y_peak_width, gyro_z_peak_width = peaks_calculator(gyro_data)
    except:
        print(acc_data.shape, target)
        print(gyro_data.shape, target)

    dictionary = {
        'acc_x_mean': x_mean,
        'acc_y_mean': y_mean,
        'acc_z_mean': z_mean,
        'acc_x_std': x_std,
        'acc_y_std': y_std,
        'acc_z_std': z_std,
        'acc_num_x_peaks': num_x_peaks,
        'acc_num_y_peaks': num_y_peaks,
        'acc_num_z_peaks': num_z_peaks,
        'acc_x_peak_prominence': x_peak_prominence,
        'acc_y_peak_prominence': y_peak_prominence,
        'acc_z_peak_prominence': z_peak_prominence,
        'acc_x_peak_width': x_peak_width,
        'acc_y_peak_width': y_peak_width,
        'acc_z_peak_width': z_peak_width,
        'gyro_x_mean': gyro_x_mean,
        'gyro_y_mean': gyro_y_mean,
        'gyro_z_mean': gyro_z_mean,
        'gyro_x_std': gyro_x_std,
        'gyro_y_std': gyro_y_std,
        'gyro_z_std': gyro_z_std,
        'gyro_num_x_peaks': gyro_num_x_peaks,
        'gyro_num_y_peaks': gyro_num_y_peaks,
        'gyro_num_z_peaks': gyro_num_z_peaks,
        'gyro_x_peak_prominence': gyro_x_peak_prominence,
        'gyro_y_peak_prominence': gyro_y_peak_prominence,
        'gyro_z_peak_prominence': gyro_z_peak_prominence,
        'gyro_x_peak_width': gyro_x_peak_width,
        'gyro_y_peak_width': gyro_y_peak_width,
        'gyro_z_peak_width': gyro_z_peak_width,
        'target': target
    }
    df = df.append(
        dictionary,
        ignore_index=True
    )
    return df


def pass_through(acc_data, gyro_data, target, df):
    acc_array = np.array(acc_data)
    ax = acc_array[:, 0]
    ay = acc_array[:, 1]
    az = acc_array[:, 2]
    gyro_array = np.array(gyro_data)
    gx = acc_array[:, 0]
    gy = acc_array[:, 1]
    gz = acc_array[:, 2]

    dictionary = {
        'ax': ax,
        'ay': ay,
        'az': az,
        'gx': gx,
        'gy': gy,
        'gz': gz,
        'target': target
    }
    df = df.append(
        dictionary,
        ignore_index=True
    )
    return df

