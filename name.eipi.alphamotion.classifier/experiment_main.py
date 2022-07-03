import time

import numpy
import pandas as pd
from data_visualization import plot_and_save_confusion_matrix
from constants import ACTIVITY_LABELS

from data_classification import learn_and_classify, confusion_matrices
from data_preprocessing import process_raw_data
import os

num_iterations = 3
training_proportions = [0.9]
slice_sizes = [100, 200, 400, 700, 1100]
sample_rate_hz = 50 #todo

sample_folder = os.path.join('erenaktas')

print("Pre-processing raw data")

timestamp_str = str(int(time.time() * 1000))
results_base_folder = os.path.join('../build/results/', timestamp_str)
os.makedirs(results_base_folder)


for slice_size in slice_sizes:
    for training_proportion in training_proportions:
        experiment_label = 'slice_size=' + str(slice_size) + ', training_fct=' + str(training_proportion)
        specific_results_folder = os.path.join(results_base_folder, experiment_label)
        os.makedirs(specific_results_folder)
        print('Running for ' + experiment_label)
        num_acc_frames_processed, num_gyro_frames_processed = process_raw_data(sample_folder, slice_size, specific_results_folder)
        df = pd.read_csv(specific_results_folder + '/final_data.csv')
        for i in range(num_iterations):
            print('Iteration ' + str(i + 1))
            learn_and_classify(df, training_proportion)

        with open(specific_results_folder + "/Setup.txt", "w") as text_file:
            print(f"Iterations: {num_iterations}", file=text_file)
            print(f"Training Proportion: {training_proportion}", file=text_file)
            print(f"Sample Rate: {sample_rate_hz}", file=text_file)
            print(f"Slice Size: {slice_size}", file=text_file)
            print(f"Total Number of Acc Frames: {num_acc_frames_processed}", file=text_file)
            print(f"Total Number of Gyro Frames: {num_gyro_frames_processed}", file=text_file)

        for name in confusion_matrices.keys():
            #print(name)
            cm_normalized = numpy.divide(confusion_matrices[name]['normalized'], num_iterations).round(2)
            #cm_normalized = confusion_matrices[name]['normalized']
            #print(confusion_matrices[name]['natural'])
            df_cm_natural = pd.DataFrame(confusion_matrices[name]['natural'], index=ACTIVITY_LABELS.values(),
                                 columns=ACTIVITY_LABELS.values())
            df_cm_normalized = pd.DataFrame(cm_normalized, index=ACTIVITY_LABELS.values(),
                                 columns=ACTIVITY_LABELS.values())
            plot_label = name + ' - ' + experiment_label
            plot_and_save_confusion_matrix(df_cm_natural, 'natural', specific_results_folder, plot_label)
            plot_and_save_confusion_matrix(df_cm_normalized, 'normalized', specific_results_folder, plot_label)

