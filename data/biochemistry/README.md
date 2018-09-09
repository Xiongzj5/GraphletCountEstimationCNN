# Empirical Graph

Here we have 3 real world biochemistry datasets, MUTAG [1], NCI1 [2], NCI109 [2], each in a HDF5 file.

## Example
You may read the data as following:
```Python
import h5py

hdf5_file = h5py.File('./NCI109.hdf5', mode='r')
graphs = hdf5_file['graph'][0:, ...] #contain graph adjacency matrices
labels = hdf5_file['label'][0:, ...] #contain 3-node, 4-node, 5-node graphlet counts

print(graphs.shape)
print(labels.shape)
```
Please refer to [HDF5 for Python documentation](http://docs.h5py.org/en/latest/index.html) for more information about how to manipulate the data.

## Reference
[1] Vishwanathan, S. V. N., Schraudolph, N. N., Kondor, R., & Borgwardt, K. M. (2010). Graph kernels. Journal of Machine Learning Research, 11(Apr), 1201-1242.

[2] Wale, N., Watson, I. A., & Karypis, G. (2008). Comparison of descriptor spaces for chemical compound retrieval and classification. Knowledge and Information Systems, 14(3), 347-375.
