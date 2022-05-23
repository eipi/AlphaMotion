import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from data_visualization import plotTsne
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from constants import FEATURE_COLUMNS
from sklearn.preprocessing import StandardScaler


param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
             ]
        }
       ]




rf = RandomForestClassifier()
dec_tre = DecisionTreeClassifier()
lr = LogisticRegression()
svc = SVC()
gb = GradientBoostingClassifier()
ml = MLPClassifier(hidden_layer_sizes=(7,))
classifiers = {
    'RandomForestClassifier': rf,  # perfect
    'DecisionTreeClassifier': dec_tre,
    'LogisticRegression': lr,
    'SVC': svc,
    'GradientBoostingClassifier': gb,  # excellent
    'MLPClassifier': ml
}
classifier_params = {
    'RandomForestClassifier': ['max_depth'],
    'DecisionTreeClassifier': ['max_depth'],
    'LogisticRegression': lr,
    'SVC': ['coef0', 'decision_function_shape'],
    'GradientBoostingClassifier': ['max_depth'],  # excellent
    'MLPClassifier': ['hidden_layer_sizes', 'alpha']
}

confusion_matrices = {}


def insert_or_multiply(name, matrix):
    if (name in confusion_matrices.keys()):
        confusion_matrices[name] = confusion_matrices[name] * matrix
    else:
        confusion_matrices[name] = matrix


def insert_or_add(name, matrix):
    if (name in confusion_matrices.keys()):
        confusion_matrices[name] = confusion_matrices[name] + matrix
    else:
        confusion_matrices[name] = matrix


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def generate_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    #print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    return cm


def visualize_classifier_confusion_matrix(cm, labels, name, x_test, y_test):
    disp = ConfusionMatrixDisplay.from_estimator(
        cm,
        x_test,
        y_test,
        display_labels=labels,
        cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.ax_.set_title(name)
    print(disp.confusion_matrix)
    plt.show()


def visualize_confusion_matrix(cm, labels, name, x_test, y_test):
    # df_cm = pd.DataFrame(cm, columns=labels, index=labels)
    df_cm = pd.DataFrame(cm)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    plt.title(label=name)
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
    plt.show()


def learn_and_classify(df, training_proportion):
    df = df.sample(frac=1).reset_index(drop=True)
    x = df[FEATURE_COLUMNS[0:9]]
    y = df.target
    x_test, x_train, y_test, y_train = train_test_split(
        x, y, test_size=training_proportion, stratify=y
    )
    labels = df.target.unique()
    scaler = StandardScaler()
    scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)
    for classifier in classifiers.keys():
        print("Training " + classifier + " classifier")
        classifiers[classifier].fit(x_train, y_train)
        print("Evaluating " + classifier + " classifier")
        cm = generate_confusion_matrix(classifiers[classifier], x_test, y_test)
        # accuracy matrix
        insert_or_add(classifier, cm)
        #visualize_classifier_confusion_matrix(classifiers[classifier], labels, classifier, x_test, y_test)
    #plotTsne(x_train, y_train, 5)


def validate_classifier_params(df):
    x = df[FEATURE_COLUMNS[0:9]]
    y = df.target
    for classifier in classifiers.keys():
        classifier_object = classifiers[classifier]
        param_grid[0]['activation'] = list(classifier_params[classifier])
        clf = GridSearchCV(classifier_object, param_grid, cv=3, scoring='accuracy')
        clf.fit(x, y)