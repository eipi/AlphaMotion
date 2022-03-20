import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def generate_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    #print(classification_report(y_test, y_pred))
    return confusion_matrix(y_test, y_pred)


def visualize_confusion_matrix(cm, labels):
    df_cm = pd.DataFrame(cm, columns=labels, index=labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
    plt.show()

