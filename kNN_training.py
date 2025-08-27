import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('telescope_data.csv')
#now we'll change class 'g' to 1 and 'h' to 0, since our computer learns faster with numbers
df['class'] = (df['class']=='g').astype(int)

train,validation,test = np.split(df.sample(frac=1), [int(0.6*len(df)),int(0.8*(len(df)))])

#now we'll scale our data in order to make it easy and precise for model to deal with the data
def scale(dataframe,oversample=False):
    X = dataframe[dataframe.columns[1:-1]].values
    y = dataframe['class'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X) #normalizing the data

    if oversample:
        ros = RandomOverSampler()
        X,y = ros.fit_resample(X,y)
    data = np.hstack((X, np.reshape(y,(-1,1))))   #since y is 1D array, so we gotta reshape it first to concatenate with X
    return data,X,y

#since the data is imbalanced, gamma class more than hedron class, so we will oversample the hedron class
#   so we've imported the imblearn library and added the oversampling parameter to the scale funtion

train, X_train, y_train = scale(train,oversample=True)

#so now train is no more a dataframe, it's a numpy matrix, and yeah the values of class 1,0 are equal now
#   print(sum(y_train==1))
#   print(sum(y_train==0))  >print these to verify

validation, X_valid, y_valid = scale(validation,oversample=False)
test, X_test, y_test = scale(test,oversample=False)

#now it's time to train our model, and we'll be doing this with kNN model, intitially taking value of k = 1
# okay, let's from sklearn.neighbors import kNeighborClassifier, assuming that we know Euclids distance formula

kNN_model = KNeighborsClassifier(n_neighbors=3)
kNN_model.fit(X_train,y_train)

#now we'll be predicting y values for the test data to asses our model
y_pred = kNN_model.predict(X_test)

#now we'll compare y_pred with y_test...it's like testing experimental values with theoritical values
# to see the report we'll do... from sklearn.metrics import Classificatio_report
print(classification_report(y_test,y_pred))


