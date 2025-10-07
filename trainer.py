import keras

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

(X_train, y_train), (X_test, y_test) = mnist.load_data()


y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test)


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

