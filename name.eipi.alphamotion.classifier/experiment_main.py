import time
from datetime import datetime
from multiprocessing import Process
from utilities import generate_current_time_string
from experiment_impl import run_experiment
import os

num_iterations = 15
training_proportions = [0.8]
slice_sizes = [25]
#sample_rates = [25,10]
sample_rates = list(range(8, 30))

now = datetime.now
timestamp_str = generate_current_time_string()
results_base_folder = os.path.join('../build/results/', timestamp_str)
os.makedirs(results_base_folder)

start = time.perf_counter()
threads = []

sample_folder = os.path.join('erenaktas')


for slice_size in slice_sizes:
    for training_proportion in training_proportions:
        if __name__ == '__main__':
            p = Process(target=run_experiment, args=(results_base_folder, sample_folder, slice_size, training_proportion, num_iterations, sample_rates))
            #t = threading.Thread(target=run_experiment, args=[slice_size, training_proportion])
            p.start()
            threads.append(p)
for thread in threads:
    thread.join()
finish = time.perf_counter()
print(f'Finished in {round(finish - start, 2)} seconds')

