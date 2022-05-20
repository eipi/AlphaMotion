import numpy
import pandas as pd
from data_classification import learn_and_classify, confusion_matrices
from data_preprocessing import extractAndProcessRawDataFiles
import os

num_iterations = 30
training_proportion = 0.5
num_samples = 61
normalization_factor = num_samples * (1 - training_proportion)
sample_rate_hz = 2 #todo

sample_folder = os.path.join('erenaktas', 'acc')

extractAndProcessRawDataFiles(sample_folder)
df = pd.read_csv('data/' + sample_folder + '/final_data.csv')

#validate_classifier_params(df)

for i in range(num_iterations):
    learn_and_classify(df, training_proportion)

for name in confusion_matrices.keys():
    print(name)
    print(confusion_matrices[name])
    print(numpy.divide(confusion_matrices[name], (num_iterations * normalization_factor)))

