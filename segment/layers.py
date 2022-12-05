import tensorflow as tf

class MaxPoolWithArgmax2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, *pool_size, 1]
        padding = padding.upper()
        strides = [1, *strides, 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding)
        argmax = tf.cast(argmax, tf.keras.backend.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

class MaxUnpool2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpool2D, self).__init__(**kwargs)
        self.size = size
    
    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = tf.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3])

        ret = tf.scatter_nd(tf.keras.backend.expand_dims(tf.keras.backend.flatten(mask)),
                            tf.keras.backend.flatten(updates),
                            [tf.keras.backend.prod(output_shape)])

        input_shape = updates.shape
        out_shape = [-1,
                     input_shape[1] * self.size[0],
                     input_shape[2] * self.size[1],
                     input_shape[3]]
        return tf.reshape(ret, out_shape)
    
    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )


class Encoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel=3, conv_layers=2, **kwargs):
        """
        filters: the number of desired output channels for each Conv2D in the block
        kernel: size of applied kernel
        conv_layers: (2 or 3) the number of convolution layers
        """
        super(Encoder, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.conv_layers = conv_layers

        if self.conv_layers == 2:
            self.conv2D_1 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2D_2 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn2 = tf.keras.layers.BatchNormalization()
        elif self.conv_layers == 3:
            self.conv2D_1 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2D_2 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.conv2D_3 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn3 = tf.keras.layers.BatchNormalization()
        else:
            print("Error in Encoder Initialization.")
    
    def call(self, inputs, **kwargs):
        conv1 = self.conv2D_1(inputs)
        conv1 = self.bn1(conv1)
        conv1 = tf.nn.leaky_relu(conv1)

        conv2 = self.conv2D_2(conv1)
        conv2 = self.bn2(conv2)
        final = tf.nn.leaky_relu(conv2)

        if self.conv_layers == 3:
            conv3 = self.conv2D_3(final)
            conv3 = self.bn3(conv3)
            final = tf.nn.leaky_relu(conv3)

        return final

class Decoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel=3, conv_layers=2, final=False, **kwargs):
        """
        filters: the number of desired output channels for each Conv2D in the block
        kernel: size of applied kernel
        conv_layers: (2 or 3) the number of convolution layers
        final: is it the final decoder in the block sequence
        """
        super(Decoder, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.conv_layers = conv_layers
        self.final = final

        if self.conv_layers == 1:
            self.conv2D_1 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
        elif self.conv_layers == 2:
            self.conv2D_1 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2D_2 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn2 = tf.keras.layers.BatchNormalization()
        elif self.conv_layers == 3:
            self.conv2D_1 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2D_2 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.conv2D_3 = tf.keras.layers.Conv2D(self.filters, [self.kernel, self.kernel], padding="same")
            self.bn3 = tf.keras.layers.BatchNormalization()
        else:
            print("Error in Decoder Initialization.")
    
    def call(self, inputs, **kwargs):
        conv1 = self.conv2D_1(inputs)
        conv1 = self.bn1(conv1)
        conv1 = tf.nn.leaky_relu(conv1)

        if not self.final:
            conv2 = self.conv2D_2(conv1)
            conv2 = self.bn2(conv2)
            final = tf.nn.leaky_relu(conv2)
        else:
            return conv1
            
        if self.conv_layers == 3:
            conv3 = self.conv2D_3(final)
            conv3 = self.bn3(conv3)
            final = tf.nn.leaky_relu(conv3)
        return final