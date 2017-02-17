from os.path import join

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from data.preprocessing import get_list_of_images


class VGG19(object):
    def __init__(self, weights_path=None, train_folder='data/train', validation_folder='data/val'):
        self.weights_path = weights_path
        self.model = self._init_model()

        if weights_path:
            self.model.load_weights(weights_path)
        else:
            self.datagen = self._init_datagen()
            self.train_folder = train_folder
            self.validation_folder = validation_folder
            self.model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

    def fit(self, batch_size=32, nb_epoch=10):

        # for epoch in xrange(nb_epoch):
        #     print 'Epoch %d' % epoch

        #     for X_train, y_train in DataGenerator():
        #         for X_batch, y_batch in self.datagen.flow(X_train, y_train, batch_size=32):
        #             loss = model.train(X_batch, y_batch)

        self.model.fit_generator(
            self._generate_data_from_folder(self.train_folder), 32,
            nb_epoch,
            verbose=1,
            callbacks=[
                TensorBoard(log_dir='./logs', write_images=True),
                ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)
            ],
            validation_data=self._generate_data_from_folder(self.validation_folder),
            nb_val_samples=32
        )

        # self.model.fit(
        #     X, y,
        #     batch_size=batch_size,
        #     nb_epoch=nb_epoch,
        #     verbose=2,
        #     callbacks=[
        #         TensorBoard(log_dir='./logs', write_images=True),
        #         ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss'),
        #         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)
        #     ],
        #     validation_split=0.2,
        #     shuffle=True,

        # )

    def evaluate(self, X, y, batch_size=32):
        return self.model.evaluate(
            X, y,
            batch_size=batch_size,
            verbose=1
        )

    def predict(self, X, batch_size=32, verbose=1):
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    def predict_proba(self, X, batch_size=32, verbose=1):
        return self.model.predict_proba(X, batch_size=batch_size, verbose=verbose)

    def _init_model(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        return model

    def _init_datagen(self):
        return ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

    def _generate_data_from_folder(self, folder_path):
        while 1:
            images = get_list_of_images(folder_path)

            for image_path in images:
                x = cv2.imread(join(folder_path, image_path))
                y = 1 if image_path.split('.')[0] == 'dog' else 0

                yield (x, y)

