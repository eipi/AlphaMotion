import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from data_visualization import plotTsne
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from constants import get_feature_columns
from sklearn.preprocessing import StandardScaler

param_grid = [
    {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
            (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
            (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,)
        ]
    }
]

classifier_params = {
    'RandomForestClassifier': ['max_depth'],
    'DecisionTreeClassifier': ['max_depth'],
    # 'LogisticRegression': lr,
    'SVC': ['coef0', 'decision_function_shape'],
    'GradientBoostingClassifier': ['max_depth'],  # excellent
    'MLPClassifier': ['hidden_layer_sizes', 'alpha']
}


def insert_or_add(confusion_matrices, name, cm_natural, cm_normalized, f_score):
    confusion_matrices_name = {}
    if name in confusion_matrices.keys():
        confusion_matrices_name['natural'] = confusion_matrices[name]['natural'] + cm_natural
        confusion_matrices_name['normalized'] = confusion_matrices[name]['normalized'] + cm_normalized
        confusion_matrices_name['fscore'] = confusion_matrices[name]['fscore'] + f_score
    else:
        confusion_matrices_name['natural'] = cm_natural
        confusion_matrices_name['normalized'] = cm_normalized
        confusion_matrices_name['fscore'] = f_score
    confusion_matrices[name] = confusion_matrices_name


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def generate_confusion_matrices(model, x_test, y_test):
    y_pred = model.predict(x_test)
    f_score = f1_score(y_test, y_pred, zero_division=1, average='weighted')
    # print(classification_report(y_test, y_pred))
    cm_natural = confusion_matrix(y_test, y_pred, normalize=None)
    cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
    return cm_natural, cm_normalized, f_score


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


def learn_and_classify(confusion_matrices, df, training_proportion):
    classifiers = {
        'RandomForestClassifier': RandomForestClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=500),
        'SVC': SVC(kernel='poly'),
        'MLPClassifier': MLPClassifier(max_iter=300)
    }
    votingClassifierSoft = VotingClassifier(estimators = [
            ('RandomForestClassifier', RandomForestClassifier()),
            ('GradientBoostingClassifier', GradientBoostingClassifier()),
            ('MLPClassifier', MLPClassifier(max_iter=300))
        ], voting="soft"
    )
    votingClassifierHard = VotingClassifier(estimators = [
            ('RandomForestClassifier', RandomForestClassifier()),
            ('GradientBoostingClassifier', GradientBoostingClassifier()),
            ('MLPClassifier', MLPClassifier(max_iter=300))
        ], voting="hard"
    )
    classifiers['VotingClassifierHard'] = votingClassifierHard
    classifiers['VotingClassifierSoft'] = votingClassifierSoft
    df = df.sample(frac=1).reset_index(drop=True)
    x = df[get_feature_columns()[0:len(get_feature_columns()) - 1]]
    y = df.target
    x_test, x_train, y_test, y_train = train_test_split(
        x, y, train_size=training_proportion, stratify=y
    )
    labels = df.target.unique()
    scaler = StandardScaler()
    scaler.fit(x_train)

    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    for classifier in classifiers.keys():
        print(classifier, end =" ")
        print("Training...", end = " ")
        classifiers[classifier].fit(x_train, y_train)
        cm_natural, cm_normalized, f_score = generate_confusion_matrices(classifiers[classifier], x_test, y_test)
        print("Done.")

        # accuracy matrix
        insert_or_add(confusion_matrices, classifier, cm_natural, cm_normalized, f_score)
        # visualize_classifier_confusion_matrix(classifiers[classifier], labels, classifier, x_test, y_test)
    # plotTsne(x_train, y_train, 5)


# def validate_classifier_params(df):
#     x = df[get_feature_columns()[0:]]
#     y = df.target
#     for classifier in classifiers.keys():
#         classifier_object = classifiers[classifier]
#         param_grid[0]['activation'] = list(classifier_params[classifier])
#         clf = GridSearchCV(classifier_object, param_grid, cv=3, scoring='accuracy')
#         clf.fit(x, y)
