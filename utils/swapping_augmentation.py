from datetime import date

import h5py
import networkx as nx
import numpy as np

if __name__ == '__main__':

    date = str(date.today())
    node_num = 28

    # open the hdf5 file
    hdf5_file = h5py.File('../data/biochemistry/MUTAG.hdf5', 'r')
    graph = hdf5_file['graph'][0:, ...]
    label = hdf5_file['label'][0:, ...]

    perm = np.arange(len(graph))
    np.random.shuffle(perm)
    graph = graph[perm]
    label = label[perm]

    p_1 = int(round(len(graph)*8/10))
    p_2 = int(round(len(graph)*9/10))
    train_graph = graph[0:p_1]
    train_label = label[0:p_1]
    validate_graph = graph[p_1:p_2]
    validate_label = label[p_1:p_2]
    test_graph = graph[p_2:len(graph)]
    test_label = label[p_2:len(graph)]

    times = 2 # how many times of reproduction

    hdf5_file_new = h5py.File('../data/biochemistry/' + date + '_MUTAG_augmentation.hdf5', mode='w')
    hdf5_file_new.create_dataset('train_graph', (p_1 * times, 1, node_num, node_num), np.int32)
    hdf5_file_new.create_dataset('train_label', (p_1 * times, 29,), np.float64)
    hdf5_file_new.create_dataset('validate_graph', (p_2 - p_1, 1, node_num, node_num), np.int32)
    hdf5_file_new.create_dataset('validate_label', (p_2 - p_1, 29,), np.float64)
    hdf5_file_new.create_dataset('test_graph', (len(graph) - p_2, 1, node_num, node_num), np.int32)
    hdf5_file_new.create_dataset('test_label', (len(graph) - p_2, 29,), np.float64)


    for i in range(validate_graph.shape[0]):
        print(i)
        A = validate_graph[i].reshape((node_num, node_num))
        G = nx.from_numpy_matrix(A)
        hdf5_file_new['validate_graph'][i, ...] = A
        hdf5_file_new['validate_label'][i, ...] = validate_label[i]

    for i in range(test_graph.shape[0]):
        print(i)
        A = test_graph[i].reshape((node_num, node_num))
        G = nx.from_numpy_matrix(A)
        hdf5_file_new['test_graph'][i, ...] = A
        hdf5_file_new['test_label'][i, ...] = test_label[i]

    counter = -1

    for i in range(train_graph.shape[0]):
        print(i)
        A = train_graph[i].reshape((node_num, node_num))
        G = nx.from_numpy_matrix(A)
        #save
        hdf5_file_new['train_graph'][i, ...] = A
        hdf5_file_new['train_label'][i, ...] = train_label[i]

        for j in range(times - 1):
            counter += 1
            A_tmp = A
            for k in range(100):
                # permute 1 pair of neighbors
                A1 = A
                a = np.random.randint(node_num)
                b = np.random.randint(node_num)
                while b == a:
                    b = np.random.randint(node_num)
                A1[[a, b]] = A1[[b, a]]
                A1 = np.transpose(A1)
                A1[[a, b]] = A1[[b, a]]
                A1 = np.transpose(A1)
                A_tmp = A1
            # save
            hdf5_file_new['train_graph'][train_graph.shape[0] + counter, ...] = A_tmp
            hdf5_file_new['train_label'][train_graph.shape[0] + counter, ...] = train_label[i]

