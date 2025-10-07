# import numpy as np
# import matplotlib.pyplot as plt
import keras

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# pre process the images

# X_train = X_train.astype(np.float32)/255
# X_test = X_test.astype(np.float32)/255

# X_train = np.expand_dims(X_train, -1)
# X_test = np.expand_dims(X_test, -1)

# convert classes to one hot vector

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test)


#error fixing
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# y_train['output_categorical'] = keras.utils.to_categorical(y_train['output_categorical'], num_classes=10)


X_train = X_train/255
X_test = X_test/255

model = Sequential()
model.add(keras.Input(shape=(28,28)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])


#earlystopping

es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1, mode='max')

#model check point

mc = ModelCheckpoint("./digit_predictor.model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

cb = [es, mc]

digit_predictor = model.fit(X_train, y_train, epochs=30, validation_split=0.3, callbacks=cb)

# Evaluate the model on test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
print('Model saved in current directory. Please run app.py')

