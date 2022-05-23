import time

import numpy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from constants import ACTIVITY_LABELS

from data_classification import learn_and_classify, confusion_matrices
from data_preprocessing import extractAndProcessRawDataFiles
import os

num_iterations = 1
training_proportion = 0.8
num_samples = 61
slice_size = 500
sample_rate_hz = 50 #todo

sample_folder = os.path.join('erenaktas', 'acc')

print("Pre-processing raw data")

timestamp_str = str(int(time.time() * 1000))
results_folder = os.path.join('../build/results/', timestamp_str)
os.makedirs(results_folder)

num_frames_processed = extractAndProcessRawDataFiles(sample_folder, slice_size, results_folder)
df = pd.read_csv(results_folder + '/final_data.csv')


def plot(cm, detail):
    plt.figure(figsize=(10, 7))
    plt.title(name)
    sns.heatmap(cm, annot=True)
    plt.savefig(results_folder + '/' + name + '_' + detail + '.png')


for i in range(num_iterations):
    learn_and_classify(df, training_proportion)


with open(results_folder + "/Setup.txt", "w") as text_file:
    print(f"Iterations: {num_iterations}", file=text_file)
    print(f"Training Proportion: {training_proportion}", file=text_file)
    print(f"Num Samples: {num_samples}", file=text_file)
    print(f"Sample Rate: {sample_rate_hz}", file=text_file)
    print(f"Slice Size: {slice_size}", file=text_file)
    print(f"Total Number of Frames: {num_frames_processed}", file=text_file)


for name in confusion_matrices.keys():
    print(name)
    #cm_normalized = numpy.divide(confusion_matrices[name]['normalized'], num_iterations).round(2)
    cm_normalized = confusion_matrices[name]['normalized']
    print(confusion_matrices[name]['natural'])
    df_cm_natural = pd.DataFrame(confusion_matrices[name]['natural'], index=ACTIVITY_LABELS.values(),
                         columns=ACTIVITY_LABELS.values())
    df_cm_normalized = pd.DataFrame(cm_normalized, index=ACTIVITY_LABELS.values(),
                         columns=ACTIVITY_LABELS.values())
    plot(df_cm_natural, 'natural')
    plot(df_cm_normalized, 'normalized')



