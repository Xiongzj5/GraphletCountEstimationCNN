# Random Graph

Here we have synthetic datasets of 2 random graph models: (1) [Random Geometric Graph](https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.generators.geometric.random_geometric_graph.html#networkx.generators.geometric.random_geometric_graph) (2) [Erdos-Renyi Graph](https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html#networkx.generators.random_graphs.erdos_renyi_graph)

## Example
You may read the data as following:
```Python
import h5py

hdf5_file = h5py.File('./RGG_radius_0.45_50node_4-clique_count.hdf5', mode='r')
# training set
train_graph = hdf5_file['train_graph'][0:, ...] #contain graph adjacency matrices
train_label = hdf5_file['train_label'][0:, ...] #contain 4-clique counts
# validation set
validate_graph = hdf5_file['validate_graph'][0:, ...] #contain graph adjacency matrices
validate_label = hdf5_file['validate_label'][0:, ...] #contain 4-clique counts
# test set
test_graph = hdf5_file['test_graph'][0:, ...] #contain graph adjacency matrices
test_label = hdf5_file['test_label'][0:, ...] #contain 4-clique counts
```
Please refer to [HDF5 for Python documentation](http://docs.h5py.org/en/latest/index.html) for more information about how to manipulate the data.
