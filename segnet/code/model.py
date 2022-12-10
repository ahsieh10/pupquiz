import tensorflow as tf
#from layers import MaxPoolWithArgmax2D, MaxUnpool2D
 
def segment(input_shape, n_labels, kernel=3, pool_size=(2, 2)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.Conv2D(64,3,strides = 2, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2D(64,3, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2D(128,3,strides = 2, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2D(128,3, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2D(256,3,strides = 2, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2D(256,3, activation = "relu",padding = "same")(x)

    x = tf.keras.layers.Conv2DTranspose(256,3, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2DTranspose(256,3,strides = 2, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2DTranspose(128,3, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2DTranspose(128,3,strides = 2, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2DTranspose(64,3, activation = "relu",padding = "same")(x)
    x = tf.keras.layers.Conv2DTranspose(256,3,strides = 2, activation = "relu",padding = "same")(x)

    outputs = tf.keras.layers.Conv2D(n_labels,3,padding = "same", activation = "softmax")(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SegNet")
 
    return model