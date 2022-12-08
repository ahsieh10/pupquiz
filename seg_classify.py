import tensorflow as tf
import segment.code.layers as layers
from segment.code.layers import MaxPoolWithArgmax2D, MaxUnpool2D
import segment.code.model

DIMS = (256, 256, 3)
N_LABELS = 2
model = segment.code.model.segment(input_shape=DIMS, n_labels=N_LABELS)
model.load_weights("/Users/pranavmahableshwarkar/CS/CSCI1470/segment_weights_35.hdf5")
print(model.summary())