import tensorflow as tf
 
def segment(input_shape, n_labels, kernel=3):
    
    # Set Up Inputs
    inputs = tf.keras.Input(shape = input_shape + (3,))
    normalize = tf.keras.layers.Rescaling(1./255)(inputs)

    # Encoding Block
    enc = tf.keras.layers.Conv2D(64,kernel,strides = 2, activation = "relu",padding = "same")(normalize)
    enc = tf.keras.layers.Conv2D(64,kernel, activation = "relu",padding = "same")(enc)
    enc = tf.keras.layers.Conv2D(128,kernel,strides = 2, activation = "relu",padding = "same")(enc)
    enc = tf.keras.layers.Conv2D(128,kernel, activation = "relu",padding = "same")(enc)
    enc = tf.keras.layers.Conv2D(256,kernel,strides = 2, activation = "relu",padding = "same")(enc)
    enc = tf.keras.layers.Conv2D(256,kernel, activation = "relu",padding = "same")(enc)

    # Decoding Block
    dec = tf.keras.layers.Conv2DTranspose(256,kernel, activation = "relu",padding = "same")(enc)
    dec = tf.keras.layers.Conv2DTranspose(256,kernel,strides = 2, activation = "relu",padding = "same")(dec)
    dec = tf.keras.layers.Conv2DTranspose(128,kernel, activation = "relu",padding = "same")(dec)
    dec = tf.keras.layers.Conv2DTranspose(128,kernel,strides = 2, activation = "relu",padding = "same")(dec)
    dec = tf.keras.layers.Conv2DTranspose(64,kernel, activation = "relu",padding = "same")(dec)
    dec = tf.keras.layers.Conv2DTranspose(256,kernel,strides = 2, activation = "relu",padding = "same")(dec)

    # Determine Logits
    outputs = tf.keras.layers.Conv2D(n_labels,kernel,padding = "same", activation = "softmax")(dec)

    # Structure Model
    model = tf.keras.Model(inputs,outputs)

    # Return Model
    return model