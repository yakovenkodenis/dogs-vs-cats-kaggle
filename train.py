from model.vgg19 import VGG19


vgg19 = VGG19(train_folder='data/train/train', validation_folder='data/val/val')

vgg19.fit(nb_epoch=1)

