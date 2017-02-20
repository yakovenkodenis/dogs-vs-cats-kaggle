from os.path import join

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from keras.utils.visualize_util import plot

from data.preprocessing import get_list_of_images


class TinyCNN(object):
    def __init__(self, weights_path=None, train_folder='data/train', validation_folder='data/val'):
        self.weights_path = weights_path
        self.model = self._init_model()
        self.datagen = self._datagen()

        if weights_path:
            self.model.load_weights(weights_path)
        else:
            self.train_folder = train_folder
            self.validation_folder = validation_folder
            self.model.compile(
                loss='binary_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy']
            )

    def fit(self, batch_size=32, nb_epoch=10):
        train_generator = self.datagen.flow_from_directory(
                self.train_folder, target_size=(224, 224),
                color_mode='rgb', class_mode='binary',
                batch_size=batch_size, shuffle=True
        )

        validation_generator = self.datagen.flow_from_directory(
            self.validation_folder, target_size=(224, 224),
            color_mode='rgb', class_mode='binary',
            batch_size=batch_size, shuffle=True
        )

        self.model.fit_generator(
            train_generator,
            samples_per_epoch=2048,
            nb_epoch=nb_epoch,
            verbose=1,
            validation_data=validation_generator,
            callbacks=[
                TensorBoard(log_dir='./logs', write_images=True),
                ModelCheckpoint(
                    filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                    save_best_only=True
                )
                # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.001)
            ],
            nb_val_samples=832,
            nb_worker=1, pickle_safe=True
        )

    def evaluate(self, X, y, batch_size=32):
        return self.model.evaluate(
            X, y,
            batch_size=batch_size,
            verbose=1
        )

    def predict(self, test_data_folder, batch_size=32, verbose=1):
        predictions = []
        for img_path in get_list_of_images(images_folder=test_data_folder):
            image = load_img(join(test_data_folder, img_path))
            x = img_to_array(image)
            x = self.datagen.random_transform(x)
            x = self.datagen.standardize(x)
            prediction = self.model.predict_proba(np.array([x]), verbose=0)
            predictions.append(prediction[0, 0])

        return predictions

    def plot_model(self, file_name):
        plot(self.model, to_file=file_name)

    def _init_model(self):
        model = Sequential()

        model.add(Convolution2D(32, 3, 3, input_shape=(224, 224, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def _datagen(self):
        return ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.2,
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

                yield x, y

