from model.tiny_cnn import TinyCNN


tinyCNN = TinyCNN(train_folder='data/train/train', validation_folder='data/val/val')

tinyCNN.fit(nb_epoch=2048, batch_size=32)

