from datetime import date

import h5py
import networkx as nx
import numpy as np

if __name__ == '__main__':
    # open the hdf5 file
    hdf5_file = h5py.File('../data/biochemistry/MUTAG.hdf5', 'r')
    graph = hdf5_file['graph'][0:, ...]
    label = hdf5_file['label'][0:, ...]
    node_num = graph.shape[2]

    # create new hdf5 file
    hdf5_file_new = h5py.File('../data/biochemistry/MUTAG_padding.hdf5', mode='w')
    hdf5_file_new.create_dataset('graph', (graph.shape[0], 1, 50, 50), np.int32)
    hdf5_file_new.create_dataset('label', (graph.shape[0], 29,), np.int32)

    for i in range(graph.shape[0]):
        print(i)
        A = graph[i].reshape((node_num, node_num))
        target = np.zeros((50, 50))
        target[:A.shape[0], :A.shape[1]] = A
        hdf5_file_new['graph'][i, ...] = target
        hdf5_file_new['label'][i, ...] = label[i]

