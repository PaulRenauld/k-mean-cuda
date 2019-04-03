# k-mean implementation on cuda
K-mean implementation on CUDA GPU with a dynamic K computation and k-mean++ computation of initial points

# Authors
- Paul Renauld
- Mathieu Chevalley

# Dataset format

The file that contains the data points follow this format. We are using CSV format. Each line can be either a parameter, the position of a centroid of a cluster, or a data point.

For a parameter, the first element of the line is the name of the paramter. The different parameters are:
 - `width`(float)
 - `height`(float)
 - `point-count`(integer)
 - `dim`(integer): dimension, by default 2
The parameters are then followed by their value.

The line for a datapoint is the list of its coordinates for all dimensions.

As for the centroid, the line starts with a `C` followed by the coordinates of the centroid. All the datapoint than then followed the declaration of the centroid are in the cluster of the said centroid, until the next centroid declaration.
