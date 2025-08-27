import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

df = pd.read_csv('telescope_data.csv')

df['class'] = (df['class']=='g').astype(int)

train,validaiton,test = np.split(df.sample(frac=1), [int(0.6*len(df)),int(0.8*len(df))])

def scale(dataframe,oversample=False):
    X = dataframe[dataframe.columns[1:-1]].values
    y = dataframe['class'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X,y = ros.fit_resample(X,y)

    return X,y

X_train,y_train = scale(train,oversample=True)
X_valid,y_valid = scale(validaiton,oversample=False)
X_test,y_test = scale(test,oversample=False)

nb_model = GaussianNB()
nb_model.fit(X_train,y_train)

y_predict = nb_model.predict(X_test)
print(classification_report(y_test,y_predict))




