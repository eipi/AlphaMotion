import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from data_classification import train_model, visualize_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from constants import FEATURE_COLUMNS
from data_preprocessing import extractAndProcessAllDataFiles
from data_classification import generate_confusion_matrix
from sklearn.preprocessing import StandardScaler

#extractAndProcessAllDataFiles()

df = pd.read_csv('data/final_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
x = df[FEATURE_COLUMNS[0:9]]
y = df.target
x_test, x_train, y_test, y_train = train_test_split(
    x, y, test_size=0.01
)
labels = df.target.unique()

scaler = StandardScaler()
scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

rf = RandomForestClassifier()
dec_tre = DecisionTreeClassifier()
lr = LogisticRegression()
svc = SVC()
gb = GradientBoostingClassifier()
ml = MLPClassifier(hidden_layer_sizes=(7,))

classifiers = {'RandomForestClassifier': rf,
               #'DecisionTreeClassifier': dec_tre,
               'LogisticRegression': lr,
               'SVC': svc,
               #'GradientBoostingClassifier': gb,
               'MLPClassifier': ml}

for classifier in classifiers.keys():
    classifiers[classifier].fit(x_train, y_train)
    visualize_confusion_matrix(generate_confusion_matrix(classifiers[classifier], x_test, y_test), labels, classifier)

