from model.tiny_cnn import TinyCNN
from utils.utils import write_predictions_to_file


tinyCNN = TinyCNN(weights_path='./weights.851-0.27.hdf5')

predictions = tinyCNN.predict('./data/test/test_folder')
write_predictions_to_file('predictions3.csv', predictions, ordering=True)

