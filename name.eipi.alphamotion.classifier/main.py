import pandas as pd
from sklearn.model_selection import train_test_split
from data_classification import train_model, visualize_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from constants import FEATURE_COLUMNS



df = pd.read_csv('data/final_data.csv')
print(df.shape)
print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
print(df.shape)
print(df.head())
x = df[FEATURE_COLUMNS[0:9]]
y = df.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=100
)
print('X tran shape:', x_train.shape)
print('X test shape:', x_test.shape)
print('y tran shape:', y_train.shape)
print('y test shape:', y_test.shape)

labels = df.target.unique()




rf = RandomForestClassifier()
rf_cm = train_model(rf, x_train, y_train, x_test, y_test)
visualize_confusion_matrix(rf)

dec_tre = DecisionTreeClassifier()
dec_tre_cm = train_model(dec_tre)
#plot_tree(dec_tre)
r = export_text(dec_tre, feature_names=FEATURE_COLUMNS[0:9])
print(r)
# result = dec_tre.predict(extractSingleFile('stairs', 5))
# print(result)

#visualize_confusion_matrix(dec_tre_cm)

# lr = LogisticRegression()
# lr_cm = train_model(lr)
# visualize_confusion_matrix(lr_cm)

# svc = SVC()
# svc_cm = train_model(svc)
# visualize_confusion_matrix(svc_cm)

# gb = GradientBoostingClassifier()
# gb_cm = train_model(gb)
# visualize_confusion_matrix(gb_cm)


