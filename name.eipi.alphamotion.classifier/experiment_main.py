import numpy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from constants import ACTIVITY_LABELS

from data_classification import learn_and_classify, confusion_matrices
from data_preprocessing import extractAndProcessRawDataFiles
import os

num_iterations = 5
training_proportion = 0.5
num_samples = 61
normalization_factor = num_samples * (1 - training_proportion)
slice_size = 250
#sample_rate_hz = 2 #todo

sample_folder = os.path.join('erenaktas', 'acc')

print("Pre-processing raw data")
extractAndProcessRawDataFiles(sample_folder, slice_size)
df = pd.read_csv('../build/final_data.csv')

#validate_classifier_params(df)

for i in range(num_iterations):
    learn_and_classify(df, training_proportion)

for name in confusion_matrices.keys():
    print(name)
    cm = numpy.divide(confusion_matrices[name], num_iterations)
    print(cm)
    df_cm = pd.DataFrame(cm, index=ACTIVITY_LABELS.values(),
                         columns=ACTIVITY_LABELS.values())
    plt.figure(figsize=(10, 7))
    plt.title(name)
    sns.heatmap(df_cm, annot=True)
    plt.show()

