import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from data_classification import train_model, visualize_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from constants import FEATURE_COLUMNS
from data_preprocessing import extractAndProcessAllDataFiles
from data_classification import generate_confusion_matrix


#extractAndProcessAllDataFiles()

df = pd.read_csv('data/final_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
x = df[FEATURE_COLUMNS[0:9]]
y = df.target
x_test, x_train, y_test, y_train = train_test_split(
    x, y, test_size=30
)
labels = df.target.unique()


rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_cf = generate_confusion_matrix(rf, x_test, y_test)
visualize_confusion_matrix(rf_cf, labels)


dec_tre = DecisionTreeClassifier()
dec_tre.fit(x_train, y_train)
dec_tre_cm = generate_confusion_matrix(dec_tre, x_test, y_test)
plot_tree(dec_tre)
# r = export_text(dec_tre, feature_names=FEATURE_COLUMNS[0:9])
# print(r)
# result = dec_tre.predict(extractSingleFile('stairs', 5))
# print(result)

#visualize_confusion_matrix(dec_tre_cm)

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_cm = generate_confusion_matrix(lr, x_test, y_test)
visualize_confusion_matrix(lr_cm, labels)

svc = SVC()
svc.fit(x_train, y_train)
svc_cm = generate_confusion_matrix(svc, x_test, y_test)
visualize_confusion_matrix(svc_cm, labels)

gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)
gb_cm = generate_confusion_matrix(gb, x_test, y_test)
visualize_confusion_matrix(gb_cm, labels)


