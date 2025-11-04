import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from xgboost import XGBClassifier


df = pd.read_csv(r"C:\Users\saaad kabir\Desktop\ML🤖\ML_1\Iris.csv")
df = df.set_index('Id')
df = df.rename(columns={'Species':'Label'})
df['Label'] = df['Label'].map({
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
})


X = df.drop('Label',axis=1)
y = df['Label']


scaler = StandardScaler()
X = scaler.fit_transform(X)




X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,stratify=y)

model = X
model = XGBClassifier(
    objective = 'multi:softmax',     #since we have 3 classes
    num_class = 3,
    learning_rate = 0.1,
    eval_metric = 'mlogloss',       #suitable for multi-class classification
    n_estimators = 100,
    max_depth = 5
)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# print("Accuracy:",accuracy_score(y_test,y_pred),"\n")
# print("Classification report: ", classification_report(y_test,y_pred))


"""

well keep in mind that in case of using xgboost, we don't have to one hot encode
the 'Label', xgb does that on his own. But make sure to use
    1. objective = 'multi:softmax'
    2. num_class = df['label'].unique().sum()
    3. eval_metric = 'mlogloss'
😇

"""



