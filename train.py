import argparse
import os
import sys
import time
from datetime import date

import h5py
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam

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


# flags
parser = argparse.ArgumentParser()
parser.add_argument('--trial', default='1')
parser.add_argument('--real', type=bool, default=False)
parser.add_argument('--epoch_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)

args = parser.parse_args()
real = args.real
trial = args.trial

# file and date
date = str(date.today())
data_file = 'data/randomGraph/ER_edge_existing_0.5_50node_4-clique_count.hdf5'
weight_file = date + '_' + trial + '_ER_edge_existing_0.5_50node_4-clique_count_weight.hdf5'
log_file = date + '_' + trial + '_ER_edge_existing_0.5_50node_4-clique_count_log.txt'

# hyper-parameters
node_num = 50
epoch_num = args.epoch_num
batch_size = args.batch_size
loss_function = 'mean_squared_error'
learning_rate = 0.0001
learning_rate_decay = 0.0
patience_1 = 500
patience_2 = 500

if real == False:
    # load data
    data = h5py.File(data_file, 'r')
    train_graph = data['train_graph'][0:, ...]
    train_label = data['train_label'][0:, ...]
    validate_graph = data['validate_graph'][0:, ...]
    validate_label = data['validate_label'][0:, ...]

    # shuffle data
    perm = np.arange(len(train_graph))
    np.random.shuffle(perm)
    train_graph = train_graph[perm]
    train_label = train_label[perm]
    perm = np.arange(len(validate_graph))
    np.random.shuffle(perm)
    validate_graph = validate_graph[perm]
    validate_label = validate_label[perm]

else:
    # load data
    data = h5py.File(data_file, 'r')
    graph = data['graph'][0:, ...]
    tmp_label = data['label'][0:, ...]
    indices = [2, 3, 4, 5, 6, 7] # only take 6 4-node graphlets counts from the labels
    label = np.empty((len(tmp_label), label_dim))
    for i in range(len(tmp_label)):
        label[i] = np.take(tmp_label[i], indices)

    # shuffle data
    perm = np.arange(len(graph))
    np.random.shuffle(perm)
    graph = graph[perm]
    label = label[perm]

    p = int(round(len(graph)*9/10))
    train_graph = graph[0:p]
    train_label = label[0:p]
    validate_graph = graph[p:len(graph)]
    validate_label = label[p:len(graph)]

# define model and compile
model = CNN(node_num)
model.compile(loss=loss_function, optimizer=Adam(lr=learning_rate, decay=learning_rate_decay))

# helper variables for main training loop
wait_1 = 0
wait_2 = 0
best_train_mae = 9999999
best_train_mape_1 = 9999999
best_train_mape_2 = 9999999
best_validate_mae = 9999999
best_validate_mape_1 = 9999999
best_validate_mape_2 = 9999999

# fit
end_epoch = 0
for epoch in range(epoch_num):
    end_epoch = epoch

    # log wall-clock time
    t = time.time()

    # single training iteration
    model.fit(train_graph, train_label, batch_size=batch_size, epochs=1, shuffle=False, verbose=0)

    # predict on whole train/validate set
    train_pred= model.predict(train_graph)
    train_pred_rounded = []
    for i in range(len(train_pred)):
        train_pred_rounded.append(round(train_pred[i][0]))

    validate_pred = model.predict(validate_graph)
    validate_pred_rounded = []
    for i in range(len(validate_pred)):
        validate_pred_rounded.append(round(validate_pred[i][0]))

    # evaluate train error
    train_mae = mae(train_label, train_pred_rounded)
    train_mape_1 = mape_1(train_label, train_pred_rounded)
    train_mape_2 = mape_2(train_label, train_pred_rounded)
    # keep best train error
    best_train_mape_1 = min(train_mape_1, best_train_mape_1)
    best_train_mape_2 = min(train_mape_2, best_train_mape_2)

    # evaluate validate error
    validate_mae = mae(validate_label, validate_pred_rounded)
    validate_mape_1 = mape_1(validate_label, validate_pred_rounded)
    validate_mape_2 = mape_2(validate_label, validate_pred_rounded)
    # keep best validate error
    best_validate_mape_1 = min(validate_mape_1, best_validate_mape_1)
    best_validate_mape_2 = min(validate_mape_2, best_validate_mape_2)

    # early stopping
    if train_mae < best_train_mae:
        best_train_mae = train_mae
        wait_1 = 0
    else:
        if wait_1 >= patience_1:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait_1 += 1

    if validate_mae < best_validate_mae:
        best_validate_mae = validate_mae
        wait_2 = 0
    else:
        if wait_2 >= patience_2:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait_2 += 1

    print('Epoch: {:05d}'.format(epoch),
          'train_mae= {:.4f}'.format(float(train_mae)),
          'train_mape_1= {:.4f}'.format(float(train_mape_1)),
          'train_mape_2= {:.4f}'.format(float(train_mape_2)),
          'val_mae= {:.4f}'.format(float(validate_mae)),
          'val_mape_1= {:.4f}'.format(float(validate_mape_1)),
          'val_mape_2= {:.4f}'.format(float(validate_mape_2)),
          'time= {:.4f}s'.format(time.time() - t))

# save weight
model.save_weights(weight_file)
print('Saved model to disk')

with open(log_file, 'w') as file:

    # log training details
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write('Dataset: %s\n' % data_file)
    file.write('Loss function: %s\n' % loss_function)
    file.write('Epoch num: %d\n' % epoch_num)
    file.write('Batch size: %d\n' % batch_size)
    file.write('Patience 1: %d\n' % patience_1)
    file.write('Patience 2: %d\n' % patience_2)
    file.write('End Epoch: %d\n' % end_epoch)
    file.write('Node num: %d\n' % node_num)
    file.write('Learning rate: %f\n' % learning_rate)
    file.write('Learning rate decay: %f\n\n' % learning_rate_decay)

    # log best train and validation error
    file.write('Best train mae: %f\n' % best_train_mae)
    file.write('Best train mape_1: %f\n' % best_train_mape_1)
    file.write('Best train mape_2: %f\n' % best_train_mape_2)

    file.write('Best validate mae during training: %f\n' % best_validate_mae)
    file.write('Best validate mape_1 during training: %f\n' % best_validate_mape_1)
    file.write('Best validate mape_2 during training: %f\n' % best_validate_mape_2)

    # predict on validate set
    validate_pred = model.predict(validate_graph)
    validate_pred_rounded = []
    for i in range(len(validate_pred)):
        validate_pred_rounded.append(round(validate_pred[i][0]))
        file.write('pre: %s\n' % validate_pred_rounded[i])
        file.write('tru: %s\n\n' % validate_label[i])

    # log validate set std and mean
    flattened_gd = validate_label.flatten()
    mean = np.mean(flattened_gd)
    file.write('mean: %f\n' % mean)
    std = np.std(flattened_gd)
    file.write('std: %f\n' % std)

    # evaluate validate error
    validate_mae = mae(validate_label, validate_pred_rounded)
    validate_mape_1 = mape_1(validate_label, validate_pred_rounded)
    validate_mape_2 = mape_2(validate_label, validate_pred_rounded)

    # log validate error
    file.write('mae error: %f\n' % validate_mae)
    file.write('mape_1 error: %f\n' % validate_mape_1)
    file.write('mape_2 error: %f\n' % validate_mape_2)

