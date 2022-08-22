import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def concat_dataset(data1, data2, axis):
    return np.concatenate((data1, data2), axis=axis)

def create_dataset(size):

    protein_1 = np.random.rand(size, 28, 28, 1) + 10
    protein_2 = np.random.rand(size, 28, 28, 1) + 20

    p1_p2 = concat_dataset(protein_1, protein_2, 3)
    p2_p1 = concat_dataset(protein_2, protein_1, 3)
    pos = concat_dataset(p1_p2, p2_p1, 0)

    protein_3 = np.random.rand(size, 28, 28, 1) + 30
    protein_4 = np.random.rand(size, 28, 28, 1) + 40

    p3_p4 = concat_dataset(protein_3, protein_4, 3)
    p4_p3 = concat_dataset(protein_4, protein_3, 3)
    neg = concat_dataset(p3_p4, p4_p3, 0)

    x = concat_dataset(pos, neg, 0)
    y = concat_dataset(np.ones(size*2), np.zeros(size*2), 0)
    return x, y


# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Random test data
# x_train, y_train = create_dataset(30000)
# x_test, y_test = create_dataset(3000)

# from sparse label to categorical
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# reshaep and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parametres
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# use functional API to build cnn layers
inputs = Input(shape=input_shape)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
# image to vector before connecting to dense layer
y = Flatten()(y)
# dropout regularization
y = Dropout(dropout)(y)
outputs = Dense(num_labels, activation='softmax')(y)

# build the model by spplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)
# network model in text
model.summary()

# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train the model with input images and lables
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=5,
          batch_size=batch_size)

# model accuracy on test dataset
score = model.evaluate(x_test,
                       y_test,
                       batch_size=batch_size,
                       verbose=0)
print('\nTest accuracyL %.1f%%' % (100.0 * score[1]))