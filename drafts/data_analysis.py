import os

from matplotlib import pyplot as plt


def extract(list, n):
    nth_items = []
    for item in list:
        if len(item) > n:
            nth_items.append(item[n])
    return nth_items


data_folder = os.path.realpath(os.path.join(os.path.join(os.path.join(os.curdir,
                                                                      "../name.eipi.alphamotion.classifier/data"), "erenaktas"), "acc"))
num_lines_per_file = {}
experiments_list = []

for file in os.listdir(data_folder):
    name = str(file).split(".txt")[0]
    experiments_list.append(name)
    num_lines_per_file[name] = sum(1 for line in open(os.path.join(data_folder, file)))

experiments_data = [str(file).split("_") for file in experiments_list]

experiments_by_users = extract(experiments_data, 2)

seconds_per_experiment = {}
for key, value in num_lines_per_file.items():
    seconds_per_experiment[key] = value / 50

times_only = list(seconds_per_experiment.values())
print(times_only)

plt.plot(times_only)
plt.ylabel("Time (s)")
plt.xlabel("Participant #")
plt.legend()
plt.show()
