import os
import csv

feature_combinations = ['all_features', 'arith_only', 'struct_only']

f1_results = [['sample_rate',
               'slice_size',
               'feature_combination',
               'RandomForestClassifier',
               'DecisionTreeClassifier',
               'GradientBoostingClassifier',
               'LogisticRegression',
               'SVC',
               'MLPClassifier',
               'VotingClassifierHard',
               'VotingClassifierSoft']]
test_base_dir = os.path.join('../build/results/', 'features_comparison')

for feature_combination in feature_combinations:
    test_dir = os.path.join('../build/results/', 'features_comparison', feature_combination)
    files = os.listdir(test_dir)

    for experiment in files:
        if experiment.startswith('slice_size'):
            sample_rate = experiment.split('sample_rate=')[1]
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
                if 'VotingClassifierHard f1-score: ' in line:
                    vpch = line.split('VotingClassifierHard f1-score: ')[1].rstrip('\n')
                if 'VotingClassifierSoft f1-score: ' in line:
                    vpcs = line.split('VotingClassifierSoft f1-score: ')[1].rstrip('\n')
                if 'Slice Size: ' in line:
                    slice_size = line.split('Slice Size: ')[1].rstrip('\n')
            f1_results.append([sample_rate, slice_size, feature_combination, rfc, dtc, gbc, lr, svc, mlpc, vpch, vpcs])

file1 = open(os.path.join(test_base_dir, 'feature_analysis.csv'), 'w', newline='')
write = csv.writer(file1)
write.writerows(f1_results)
