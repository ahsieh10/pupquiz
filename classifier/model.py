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
    model = tf.keras.layers.Sequential()
    model.add(tf.keras.Input(input_shape=DIMS))

    # 1st conv block
    model.add(tf.keras.layers.Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    # 2nd conv block
    model.add(tf.keras.layers.Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    # 3rd conv block
    model.add(tf.keras.layers.Conv2D(120, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(tf.keras.layers.BatchNormalization())
    # ANN block
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=200, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    # output layer
    model.add(tf.keras.layers.Dense(units=BREEDS, activation='softmax'))

    # compile model
    # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # fit on data for 30 epochs
    # model.fit_generator(train, epochs=30, validation_data=val)
    return model

