import numpy as np
import gzip

data_path = './data/'
train_data_list = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']
test_data_list = ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

with gzip.open(data_path + train_data_list[1], 'rb') as f:
    y_train = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open(data_path + train_data_list[0], 'rb') as f:
    x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

with gzip.open(data_path + test_data_list[1], 'rb') as f:
    y_test = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open(data_path + test_data_list[0], 'rb') as f:
    x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
np.save(data_path + 'train_images.npy', x_train)
np.save(data_path + 'train_labels.npy', y_train)
np.save(data_path + 'test_images.npy', x_test)
np.save(data_path + 'test_labels.npy', y_test)