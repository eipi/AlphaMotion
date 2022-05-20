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
def feature_engineer(action, target, df):
    try:
        x_mean, y_mean, z_mean = mean_calculator(action)
        x_std, y_std, z_std = std_calculator(action)
        num_x_peaks, num_y_peaks, num_z_peaks, x_peak_prominence, y_peak_prominence, z_peak_prominence,\
            x_peak_width, y_peak_width, z_peak_width = peaks_calculator(action)
    except:
        print(action.shape, target)

    dictionary = {
        'x_mean': x_mean,
        'y_mean': y_mean,
        'z_mean': z_mean,
        'x_std': x_std,
        'y_std': y_std,
        'z_std': z_std,
        'num_x_peaks': num_x_peaks,
        'num_y_peaks': num_y_peaks,
        'num_z_peaks': num_z_peaks,
        'x_peak_prominence': x_peak_prominence,
        'y_peak_prominence': y_peak_prominence,
        'z_peak_prominence': z_peak_prominence,
        'x_peak_width': x_peak_width,
        'y_peak_width': y_peak_width,
        'z_peak_width': z_peak_width,
        'target': target
    }
    df = df.append(
        dictionary,
        ignore_index=True
    )
    return df

