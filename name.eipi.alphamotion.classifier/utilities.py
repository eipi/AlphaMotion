from math import floor, ceil

import numpy as np
from constants import default_sample_rate_hz
from scipy.signal import find_peaks, peak_prominences, peak_widths
from datetime import datetime


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


    if num_x_peaks != 0:
        x_peak_prominences = np.mean(peak_prominences(vector_x, x_peaks))
        x_peak_widths = np.mean(peak_widths(vector_x, x_peaks))
    else:
        x_peak_prominences = 0
        x_peak_widths = 0

    if num_y_peaks != 0:
        y_peak_prominences = np.mean(peak_prominences(vector_y, y_peaks))
        y_peak_widths = np.mean(peak_widths(vector_y, y_peaks))
    else:
        y_peak_prominences = 0
        y_peak_widths = 0

    if num_z_peaks != 0:
        z_peak_prominences = np.mean(peak_prominences(vector_z, z_peaks))
        z_peak_widths = np.mean(peak_widths(vector_z, z_peaks))
    else:
        z_peak_prominences = 0
        z_peak_widths = 0

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
        x_num_peaks, y_num_peaks, z_num_peaks, x_peak_prominence, y_peak_prominence, z_peak_prominence,\
            x_peak_width, y_peak_width, z_peak_width = peaks_calculator(acc_data)
        gyro_x_mean, gyro_y_mean, gyro_z_mean = mean_calculator(gyro_data)
        gyro_x_std, gyro_y_std, gyro_z_std = std_calculator(gyro_data)
        gyro_x_num_peaks, gyro_y_num_peaks, gyro_z_num_peaks, gyro_x_peak_prominence, gyro_y_peak_prominence, gyro_z_peak_prominence,\
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
        'acc_x_num_peaks': x_num_peaks,
        'acc_y_num_peaks': y_num_peaks,
        'acc_z_num_peaks': z_num_peaks,
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
        'gyro_x_num_peaks': gyro_x_num_peaks,
        'gyro_y_num_peaks': gyro_y_num_peaks,
        'gyro_z_num_peaks': gyro_z_num_peaks,
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


def generate_current_time_string():
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y_%H%M%S")
    return date_time

original_sample_rate_hz = 50


def resample_list(list, samplerate):
    if samplerate > default_sample_rate_hz:
        print('Upsampling not supported')
        return
    resampled_array = []
    num_samples = len(list)
    time = num_samples / default_sample_rate_hz

    num_resamples = time * samplerate
    period = num_samples / num_resamples

    i = 1
    while i <= len(list) - 1:
        time_floor = floor(i)
        time_ceil = ceil(i)
        upper_weight = i - time_floor
        lower_weight = 1 - upper_weight
        if time_ceil < len(list):
            resampled_array.append(list[time_floor] * lower_weight + list[time_ceil] * upper_weight)
        else:
            resampled_array.append(list[time_floor])
        i = i + period
    return resampled_array

