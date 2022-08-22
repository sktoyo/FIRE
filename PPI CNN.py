import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization, Activation, MaxPool1D
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, MaxPool1D, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model

# input feature shape: [(None, (2, 512, 20)], 10000 positive pairs, 10000 negative pairs, 2000 test pairs(pos/neg)
def concat_dataset(data1, data2, axis):
    return np.concatenate((data1, data2), axis=axis)

def create_dataset(size):
    protein_1 = np.random.rand(size, 1, 512, 20) + 10
    protein_2 = np.random.rand(size, 1, 512, 20) + 20

    p1_p2 = concat_dataset(protein_1, protein_2, 1)
    p2_p1 = concat_dataset(protein_2, protein_1, 1)
    pos = concat_dataset(p1_p2, p2_p1, 0)

    protein_3 = np.random.rand(size, 1, 512, 20) + 30
    protein_4 = np.random.rand(size, 1, 512, 20) + 40

    p3_p4 = concat_dataset(protein_3, protein_4, 1)
    p4_p3 = concat_dataset(protein_4, protein_3, 1)
    neg = concat_dataset(p3_p4, p4_p3, 0)

    x = concat_dataset(pos, neg, 0)
    y = concat_dataset(np.ones(size*2), np.zeros(size*2), 0)
    return x, y


def create_dataset_constant(size):
    protein_1 = np.zeros((size, 1, 512, 20)) + 10
    protein_2 = np.zeros((size, 1, 512, 20)) + 20

    p1_p2 = concat_dataset(protein_1, protein_2, 1)
    p2_p1 = concat_dataset(protein_2, protein_1, 1)
    pos = concat_dataset(p1_p2, p2_p1, 0)

    protein_3 = np.zeros((size, 1, 512, 20)) + 30
    protein_4 = np.zeros((size, 1, 512, 20)) + 40

    p3_p4 = concat_dataset(protein_3, protein_4, 1)
    p4_p3 = concat_dataset(protein_4, protein_3, 1)
    neg = concat_dataset(p3_p4, p4_p3, 0)

    x = concat_dataset(pos, neg, 0)
    y = concat_dataset(np.ones(size*2), np.zeros(size*2), 0)
    return x, y

x_train, y_train = create_dataset(1000)
x_test, y_test = create_dataset(100)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# visualize data distribution
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
x_train_vis = np.concatenate((x_train[:50], x_train[-50:]), axis=0)
x_train_vis = x_train_vis.reshape((-1, 2*512*20))
x_test_vis = np.concatenate((x_test[:50], x_test[-50:]), axis=0)
x_test_vis = x_train_vis.reshape((-1, 2*512*20))
total_2d = tsne.fit_transform(concat_dataset(x_train_vis, x_test_vis,0))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
plt.scatter(total_2d[:50][:,0], total_2d[:50][:,1], c='r', label='train_neg')
plt.scatter(total_2d[50:100][:,0], total_2d[50:100][:,1], c='g', label='train_pos')
plt.scatter(total_2d[100:150][:,0], total_2d[100:150][:,1], c='b', label='test_neg')
plt.scatter(total_2d[150:][:,0], total_2d[150:][:,1], c='y', label='test_pos')
plt.legend()
plt.show()

# from sparse label to categorical
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# resize and normalize
x_train = np.reshape(x_train, [-1, 2, 512, 20])
x_test = np.reshape(x_test, [-1, 2, 512, 20])
# x_train = np.reshape(x_train, [-1, 28, 28, 1]) # MNIST test
# x_test = np.reshape(x_test, [-1, 28, 28, 1]) # MNIST test
x_max = x_train.max()
x_min = x_train.min()
x_train = (x_train.astype('float32') - x_min) / (x_max - x_min)
x_test = (x_test.astype('float32') - x_min) / (x_max - x_min)

print(x_train.shape)
print(x_test.shape)
# network parametres
input_shape = (2, 512, 20)
# input_shape = (28, 28, 1) # MNIST test
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# network setting
def conv2D_model():
    inputs = Input(shape=input_shape)
    y = Conv2D(filters,
               kernel_size=(2, kernel_size),
               activation='relu', padding="same")(inputs)
    y = MaxPool2D(padding="same", strides=(1, 1))(y)
    y = Conv2D(filters,
               kernel_size=(2, kernel_size),
               activation='relu', padding="same")(y)
    y = MaxPool2D(padding="same", strides=(1, 1))(y)
    y = Conv2D(filters,
               kernel_size=(2, kernel_size),
               activation='relu', padding="same")(y)
    # image to vector before connecting to dense layer
    y = Flatten()(y)
    # dropout regularization
    y = Dropout(dropout)(y)
    outputs = Dense(num_labels, activation='softmax')(y)
    return Model(inputs=inputs, outputs=outputs)

def conv1D_model():
    inputs = Input(shape=input_shape)
    y = Conv2D(filters,
               kernel_size=(2, kernel_size),
               strides=2,
               )(inputs)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Reshape((int(input_shape[0]/2*(input_shape[1]/2-1)), filters))(y)
    y = MaxPool1D()(y)
    y = Conv1D(filters*2,
               kernel_size=kernel_size,
               strides=2)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = MaxPool1D()(y)
    # y = Conv1D(filters,
    #            kernel_size=kernel_size,
    #            activation='relu')(y)

    # image to vector before connecting to dense layer
    y = Flatten()(y)
    # dropout regularization
    y = Dropout(dropout)(y)
    outputs = Dense(num_labels, activation='softmax')(y)
    return Model(inputs=inputs, outputs=outputs)

# build the model by spplying inputs/outputs
model = conv1D_model()
# network model in text
model.summary()
plot_model(model, to_file='conv1d_toy_data.png', show_shapes=True)

# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train the model with input images and labels
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

print(model(x_test[:10]))
print(model(x_test[-10:]))
print(model(x_train[:10]))
print(model(x_train[-10:]))