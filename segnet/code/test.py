import numpy as np
import tensorflow as tf
from keras.layers import Input
from layers import MaxPoolWithArgmax2D, MaxUnpool2D

def test_max_unpooling():
    # GIVEN we have some dummy test data which is 1-4 arranged as a 2 x 2
    data = np.asarray([[
        [1, 2],
        [3, 4]]], np.float32)
    data = np.expand_dims(data, axis=-1)
    tensor = tf.convert_to_tensor(data, np.float32)

    # WHEN we supply this data to our custom layers
    inp = Input(shape=(2, 2, 1))
    pool_1, mask_1 = MaxPoolWithArgmax2D()(inp)
    out = MaxUnpool2D()([pool_1, mask_1])
    model = tf.keras.Model(inp, out)
    result = model.predict([tensor])

    # THEN the output should be a sparse version of our input (only the maximum argument is retained)
    assert result.tolist()[0] == [[[0.0], [0.0]], [[0.0], [4.0]]]