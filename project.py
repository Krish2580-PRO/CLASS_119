import pandas as pa
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image
import pydotplus

# ------------------------------------------------------------------------------------------------------------------------------------------

col_names = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Survived']

data = pa.read_csv("diabetes.csv" , names = col_names).iloc[1:]

# ------------------------------------------------------------------------------------------------------------------------------------------

features = ['PassengerId','Pclass','Sex','Age','SibSp','Parch']

X = data[features]
Y = data.Outcome

X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size=0.3)

classifier = DecisionTreeClassifier()

classifier = classifier.fit(X_train , Y_train)

Y_pred = classifier.predict(X_test)

print("Accuracy : " , accuracy_score(Y_test,Y_pred))

# -----------------------------------------------------------------------

d_data = StringIO()

export_graphviz(classifier , out_file = d_data , filled=True , rounded = True , special_characters=True , feature_names = features, class_names = ['0','1'])

print(d_data.getvalue())

# ---------------------------------------------------------------------------------


img = pydotplus.graph_from_dot_data(d_data.getvalue())

img.write_png('survived.png')
Image(img.create_png())

# -----------------------------------------------------------------------------------

classifier = DecisionTreeClassifier(max_depth = 3)

classifier = classifier.fit(X_train , Y_train)

Y_pred = classifier.predict(X_test)

print("Accuracy : ", accuracy_score(Y_test, Y_pred))



d_data = StringIO()

export_graphviz(classifier , out_file = d_data , filled=True , rounded = True , special_characters=True , feature_names = features, class_names = ['0','1'])

print(d_data.getvalue())

# ---------------------------------------------------------------------------------


img = pydotplus.graph_from_dot_data(d_data.getvalue())

img.write_png('survived.png')
Image(img.create_png())



