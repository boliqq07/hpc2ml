For some general function.
such as for manipulate function for data.
It could be used in

1. data preprocessing part (mainly numpy function);
2. data transform part (tensor function, and be warped as Transformer class);
3. network part (tensor function)

The 2 part are used but not concern gradient.
The 3 part are used gradient.

We strongly recommend in 1 part, do just raw data preparation.

We strongly recommend in 2 part, just get edge_index, edge_weight, edge_attr (if regress_force not concerned), else
do just get edge_index (if regress_force concerned).

For speed up, we suggest get edge_index, then get edge_weight, edge_attr in network.
