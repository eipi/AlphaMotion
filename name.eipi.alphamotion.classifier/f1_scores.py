import os
import csv

#test_dir = os.path.join('../build/results/','07252022_154200')
#test_dir = os.path.join('../build/results/','07252022_230652')
#test_dir = os.path.join('../build/results/','07252022_231450')
#test_dir = os.path.join('../build/results/','07252022_232335')
#test_dir = os.path.join('../build/results/','07262022_000213')
#test_dir = os.path.join('../build/results/','07302022_235740_1')
test_dir = os.path.join('../build/results/', '08012022_161252')
files = os.listdir(test_dir)
f1_results = []
f1_results.append(['sample_rate',
                   'RandomForestClassifier',
                   'DecisionTreeClassifier',
                   'GradientBoostingClassifier',
                   'LogisticRegression',
                   'SVC',
                   'MLPClassifier'])
for experiment in files:
    if (experiment.startswith('slice_size')):
        sr = experiment.split('sample_rate=')[1]
        # Using readlines()
        details_file = open(os.path.join(test_dir, experiment, 'Setup.txt'), 'r')
        details_file_lines = details_file.readlines()

        for line in details_file_lines:
            if 'RandomForestClassifier f1-score: ' in line:
                rfc = line.split('RandomForestClassifier f1-score: ')[1].rstrip('\n')
            if 'DecisionTreeClassifier f1-score: ' in line:
                dtc = line.split('DecisionTreeClassifier f1-score: ')[1].rstrip('\n')
            if 'GradientBoostingClassifier f1-score: ' in line:
                gbc = line.split('GradientBoostingClassifier f1-score: ')[1].rstrip('\n')
            if 'LogisticRegression f1-score: ' in line:
                lr = line.split('LogisticRegression f1-score: ')[1].rstrip('\n')
            if 'SVC f1-score: ' in line:
                svc = line.split('SVC f1-score: ')[1].rstrip('\n')
            if 'MLPClassifier f1-score: ' in line:
                mlpc = line.split('MLPClassifier f1-score: ')[1].rstrip('\n')
        f1_results.append([sr, rfc, dtc, gbc, lr, svc, mlpc])

    print(f1_results)
    file1 = open(os.path.join(test_dir, 'f1_analysis.csv'), 'w', newline ='')
    write = csv.writer(file1)
    write.writerows(f1_results)
