import random
from os import listdir, rename
from os.path import isfile, join

import numpy as np
from PIL import Image


WIDTH = 224
HEIGHT = 224

def preprocess(images_folder='./train/train'):
    images = get_list_of_images(images_folder)

    global WIDTH
    global HEIGHT

    for image_path in images:
        rescale_image(image_path, images_folder, (WIDTH, HEIGHT))


def get_list_of_images(images_folder='./train/train'):
    return [f for f in listdir(images_folder) if isfile(join(images_folder, f))]


def get_mean_image_size(images, images_folder='./train/train'):
    sizes = []
    for image_path in images:
        image = Image.open(join(images_folder, image_path))
        sizes.append(image.size)

    sizes = np.array(sizes)
    meanW, meanH = np.mean(sizes[:, 0]), np.mean(sizes[:, 1])

    return meanW, meanH


def rescale_image(image_path, images_folder, size):
    source_path = join(images_folder, image_path)
    image = Image.open(source_path)
    image = image.resize(size, Image.ANTIALIAS)
    image.save(source_path, 'JPEG')


def create_validation_set(images_folder='./train/train', size=0.2):
    images = get_list_of_images(images_folder)
    val_size = int(size * len(images))
    val_set = [images[i] for i in sorted(random.sample(xrange(len(images)), val_size))]

    for image_path in val_set:
        rename(join(images_folder, image_path), join('./val/val', image_path))


# def grayscale_images(images_folder='./train/train'):
    
