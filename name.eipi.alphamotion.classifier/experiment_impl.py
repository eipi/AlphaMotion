import os

import numpy
import pandas as pd
from data_visualization import plot_and_save_confusion_matrix
from constants import ACTIVITY_LABELS
from data_classification import learn_and_classify
from data_preprocessing import process_raw_data


def run_experiment(results_base_folder, sample_folder, slice_size, training_proportion, num_iterations, sample_rates):
    for sample_rate in sample_rates:
        experiment_label = 'slice_size=' + str(slice_size) + ', training_fct=' + str(
            training_proportion) + ', sample_rate=' + str(sample_rate)
        specific_results_folder = os.path.join(results_base_folder, experiment_label)
        os.makedirs(specific_results_folder)
        print('Running for ' + experiment_label)
        num_acc_frames_processed, num_gyro_frames_processed = process_raw_data(sample_folder, specific_results_folder,
                                                                               slice_size, sample_rate)
        df = pd.read_csv(specific_results_folder + '/final_data.csv')
        confusion_matrices = {}
        for i in range(num_iterations):
            print('Iteration ' + str(i + 1))
            learn_and_classify(confusion_matrices, df, training_proportion)

        with open(specific_results_folder + "/Setup.txt", "w") as text_file:
            print(f"Iterations: {num_iterations}", file=text_file)
            print(f"Training Proportion: {training_proportion}", file=text_file)
            print(f"Sample Rate: {sample_rate}", file=text_file)
            print(f"Slice Size: {slice_size}", file=text_file)
            print(f"Total Number of Acc Frames: {num_acc_frames_processed}", file=text_file)
            print(f"Total Number of Gyro Frames: {num_gyro_frames_processed}", file=text_file)
        text_file.close()
        for name in confusion_matrices.keys():
            cm_normalized = numpy.divide(confusion_matrices[name]['normalized'], num_iterations).round(2)
            df_cm_natural = pd.DataFrame(confusion_matrices[name]['natural'], index=ACTIVITY_LABELS.values(),
                                         columns=ACTIVITY_LABELS.values())
            df_cm_normalized = pd.DataFrame(cm_normalized, index=ACTIVITY_LABELS.values(),
                                            columns=ACTIVITY_LABELS.values())
            f1_score = numpy.divide(confusion_matrices[name]['fscore'], num_iterations).round(2)
            plot_label = name + ' - ' + experiment_label
            plot_and_save_confusion_matrix(df_cm_natural, 'natural', specific_results_folder, plot_label)
            plot_and_save_confusion_matrix(df_cm_normalized, 'normalized', specific_results_folder, plot_label)
            print(name + ' f1-score = ' + str(f1_score))
            with open(specific_results_folder + "/Setup.txt", "a", ) as text_file:
                print(f"{name} f1-score: {f1_score}", file=text_file)

