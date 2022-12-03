import tensorflow as tf
from layers import MaxPoolWithArgmax2D, MaxUnpool2D, Encoder, Decoder

def segment(input_shape, n_labels, kernel=3, pool_size=(2, 2)):
    inputs = tf.keras.Iput(shape=input_shape)

    # Encoder 1
    enc1 = Encoder(filters=64, kernel=kernel)(inputs)
    pool1, mask1 = MaxPoolWithArgmax2D(pool_size)(enc1)

    # Encoder 2
    enc2 = Encoder(filters=128, kernel=kernel)(pool1)
    pool2, mask2 = MaxPoolWithArgmax2D(pool_size)(enc2)

    # Encoder 3
    enc3 = Encoder(filters=256, kernel=kernel, conv_layers=3)(pool2)
    pool3, mask3 = MaxPoolWithArgmax2D(pool_size)(enc3)

    # Encoder 4
    enc4 = Encoder(filters=512, kernel=kernel, conv_layers=3)(pool3)
    pool4, mask4 = MaxPoolWithArgmax2D(pool_size)(enc4)

    # Encoder 5
    enc5 = Encoder(filters=512, kernel=kernel, conv_layers=3)(pool4)
    pool5, mask5 = MaxPoolWithArgmax2D(pool_size)(enc5)

    print("Encoding Completed... Beginning Decoding.")

    # Decoder 1
    unpool1 = MaxUnpool2D(pool_size)([pool5, mask5])
    decode1 = Decoder(filters=512, kernel=kernel, conv_layers=3)(unpool1)
    
    # Decoder 2
    unpool2 = MaxUnpool2D(pool_size)([decode1, mask4])
    decode2 = Decoder(filters=512, kernel=kernel, conv_layers=3)(unpool2)

    # Decoder 3
    unpool3 = MaxUnpool2D(pool_size)([decode2, mask3])
    decode3 = Decoder(filters=512, kernel=kernel, conv_layers=3)(unpool3)

    # Decoder 4
    unpool4 = MaxUnpool2D(pool_size)([decode3, mask2])
    decode4 = Decoder(filters=512, kernel=kernel)(unpool4)

    # Decoder 5
    unpool5 = MaxUnpool2D(pool_size)([decode4, mask1])
    decode5 = Decoder(filters=512, kernel=kernel, final=True)(unpool5)

    conv = tf.keras.layers.Conv2D(n_labels, (1, 1), padding="valid")(decode5)
    norm = tf.keras.layers.BatchNormalization()(conv)
    out = tf.reshape((input_shape[0] * input_shape[1], n_labels),
                    input_shape=(input_shape[0], input_shape[1], n_labels))(norm)

    logits = tf.keras.layers.Softmax(out)
    print("Decoding Complete!")
    
    model = tf.keras.Model(inputs=inputs, outputs=logits, name="Segment Dog")