import time

from multiprocessing import Process
from utilities import generate_current_time_string
from experiment_impl import run_experiment
import os

num_iterations = 30
training_proportions = [0.6]
slice_sizes = [50]
sample_rates = [25,26,27,28]
experiment_name = 'all_features'


def create_base_results_folder():
    global results_base_folder
    timestamp_str = generate_current_time_string()
    results_base_folder = os.path.join('../build/results/', timestamp_str, experiment_name)
    os.makedirs(results_base_folder, exist_ok=True)
    return results_base_folder


results_base_folder = create_base_results_folder()


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

