from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten, Dense
from keras.models import Sequential


def CNN(node_num=50):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1', input_shape=(1, node_num, node_num)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv2'))
    model.add(Flatten())
    model.add(Dense(1, activation='relu', name='dense1'))
    model.summary()

    return model
