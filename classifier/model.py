import tensorflow as tf

def mnetV2(DIMS, BREEDS):
    mnetV2 = tf.keras.applications.MobileNetV2(input_shape=DIMS,
                                               include_top=False,
                                               weights='imagenet')
    mnetV2.trainable = False
    
    model = tf.keras.Sequential([
        mnetV2,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(BREEDS, activation='softmax')
    ])

    return model

def naive_class(DIMS, BREEDS):
    
    return

