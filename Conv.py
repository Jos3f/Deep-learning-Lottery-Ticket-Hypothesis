from keras.datasets import cifar10
import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

class Conv:
    def __init__(self, model_name, use_dropout=False, train=True):
        self.K = 10

        self.learning_rate = 0.0003
        self.batch_size = 60

        iterations = 1000

        self.model = self.build_model(model_name, use_dropout)
        if train:
            self.model = self.train(self.model, iterations)
            self.model.save_weights(model_name + '.h5')
        else:
            self.model.load_weights(model_name + '.h5')


    def build_model(self, model_name, use_dropout):

        assert model_name in ['conv-2', 'conv-4', 'conv-6'], "Use Conv-2, Conv-4 or Conv-6 model"

        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=x_train[0].shape, kernel_initializer='glorot_normal'))
        if use_dropout: model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
        if use_dropout: model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))

        if model_name in ['conv-2', 'conv-4']:
            model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
            if use_dropout: model.add(Dropout(0.5))
            model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
            if use_dropout: model.add(Dropout(0.5))
            model.add(MaxPooling2D((2, 2)))

        if model_name == 'conv-6':
            model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
            if use_dropout: model.add(Dropout(0.5))
            model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
            if use_dropout: model.add(Dropout(0.5))
            model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
        if use_dropout: model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
        if use_dropout: model.add(Dropout(0.5))
        model.add(Dense(self.K, activation='softmax', kernel_initializer='glorot_normal'))
        return model


    def predict(self,x, batch_size=50):

        return self.model.predict(x, batch_size)

    def train(self, model, iterations):

        adam = optimizers.adam(self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # Here we save the initial weights that we can use later. Also here we would add code to apply masks to the weights
        initial_weights = model.get_weights()

        # Apply masks to prune percentage of weights here

        model.set_weights(initial_weights)


        epochs = int(iterations * self.batch_size / x_train.shape[0])
        model.fit(x_train, y_train, self.batch_size, epochs)

        return model

if __name__ == '__main__':

    K = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, K)
    y_test = keras.utils.to_categorical(y_test, K)

    model = Conv('conv-6')

    predicted_x = model.predict(x_test)
    correct = np.argmax(predicted_x, 1) == np.argmax(y_test, 1)

    acc = sum(correct) / len(correct)
    print("Test Acc: ", acc)
