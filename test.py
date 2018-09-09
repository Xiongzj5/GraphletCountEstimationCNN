import os
import sys
from datetime import date

import h5py
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.47
set_session(tf.Session(config=config))

sys.path.insert(0, './')

from NNModel.CNN import CNN
from utils.mae import mae
from utils.mape_1 import mape_1
from utils.mape_2 import mape_2
from utils.under_over import under_over
from sklearn.metrics import r2_score


date = str(date.today())
log_file = date + '_ER_edge_existing_0.5_50node_4-clique_count_test_result.txt'
file = open(log_file, 'w')

node_num = 50
model = CNN(node_num)
model.load_weights('2018-09-09_1_ER_edge_existing_0.5_50node_4-clique_count_weight.hdf5') # put the file name of your CNN weights here

hdf5_file = h5py.File('data/randomGraph/ER_edge_existing_0.5_50node_4-clique_count.hdf5', 'r')
test_graph = hdf5_file['test_graph'][0:, ...]
test_label = hdf5_file['test_label'][0:, ...]

# predict on whole test set
test_pred = model.predict(test_graph)
test_pred_rounded = []
for i in range(len(test_pred)):
    test_pred_rounded.append(round(test_pred[i][0]))
    file.write('pre: %s\n' % test_pred_rounded[i])
    file.write('tru: %s\n\n' % test_label[i])

# pred sum
s = np.sum(test_pred_rounded)
file.write('pred sum: %f\n' % s)
ts = np.sum(test_label)
file.write('truth sum: %f\n' % ts)

# under predict, over predict
over_num, under_num, over_loss, under_loss = under_over(test_label, test_pred_rounded)
file.write('number of over prediction samples: %f\n' % over_num)
file.write('number of under prediction samples: %f\n' % under_num)
file.write('average loss of over predictions: %f\n' % over_loss)
file.write('average loss of under predictions: %f\n' % under_loss)

# evaluate train error
test_mae = mae(test_label, test_pred_rounded)
test_mape_1 = mape_1(test_label, test_pred_rounded)
test_mape_2 = mape_2(test_label, test_pred_rounded)


# log validate set std and mean
flattened_gd = test_label.flatten()
mean = np.mean(flattened_gd)
file.write('mean: %f\n' % mean)
std = np.std(flattened_gd)
file.write('std: %f\n' % std)

# log test error
file.write('mae error: %f\n' % test_mae)
file.write('mpae_1 error: %f\n' % test_mape_1)
file.write('mpae_2 error: %f\n' % test_mape_2)
r2 = r2_score(test_label, test_pred_rounded)
file.write('R^2 score (best score is 1.0, score can be negative): %f\n' % r2)


