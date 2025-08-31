import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv('telescope_data.csv')
df = df.set_index('index')
df['class'] = (df['class']=='g').astype(int)

train,val,test = np.split(df.sample(frac=1), [int(0.6*len(df)),int(0.8*len(df))])


def scale(dataframe,oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X,y = ros.fit_resample(X,y)

  return X,y

X_train, y_train = scale(train,oversample=True)
X_val, y_val = scale(val)
X_test, y_test = scale(test)

nn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Binary cross_entropy')
  plt.legend()
  plt.grid(True)
  plt.show()

def plot_acuracy(history):
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True)
  plt.show()

nn_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = nn_model.fit(
    X_train,y_train,
    validation_split = 0.2,
    batch_size=32,
    epochs=100,
    verbose=1
)
plot_loss(history)
plot_acuracy(history)