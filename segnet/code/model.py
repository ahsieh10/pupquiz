import tensorflow as tf
from layers import MaxPoolWithArgmax2D, MaxUnpool2D
 
def segment(input_shape, n_labels, kernel=3, pool_size=(2, 2)):
   inputs = tf.keras.layers.Input(shape=input_shape)
 
   conv_1 = tf.keras.layers.Conv2D(64, (kernel, kernel), padding="same")(inputs)
   conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
   conv_1 = tf.keras.layers.Activation("relu")(conv_1)
   conv_2 = tf.keras.layers.Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
   conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
   conv_2 = tf.keras.layers.Activation("relu")(conv_2)
 
   pool_1, mask_1 = MaxPoolWithArgmax2D(pool_size)(conv_2)
 
   conv_3 = tf.keras.layers.Conv2D(128, (kernel, kernel), padding="same")(pool_1)
   conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
   conv_3 = tf.keras.layers.Activation("relu")(conv_3)
   conv_4 = tf.keras.layers.Conv2D(128, (kernel, kernel), padding="same")(conv_3)
   conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
   conv_4 = tf.keras.layers.Activation("relu")(conv_4)
 
   pool_2, mask_2 = MaxPoolWithArgmax2D(pool_size)(conv_4)
 
   conv_5 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(pool_2)
   conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
   conv_5 = tf.keras.layers.Activation("relu")(conv_5)
   conv_6 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(conv_5)
   conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
   conv_6 = tf.keras.layers.Activation("relu")(conv_6)
   conv_7 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(conv_6)
   conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
   conv_7 = tf.keras.layers.Activation("relu")(conv_7)
 
   pool_3, mask_3 = MaxPoolWithArgmax2D(pool_size)(conv_7)
 
   conv_8 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(pool_3)
   conv_8 = tf.keras.layers.BatchNormalization()(conv_8)
   conv_8 = tf.keras.layers.Activation("relu")(conv_8)
   conv_9 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_8)
   conv_9 = tf.keras.layers.BatchNormalization()(conv_9)
   conv_9 = tf.keras.layers.Activation("relu")(conv_9)
   conv_10 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_9)
   conv_10 = tf.keras.layers.BatchNormalization()(conv_10)
   conv_10 = tf.keras.layers.Activation("relu")(conv_10)
 
   pool_4, mask_4 = MaxPoolWithArgmax2D(pool_size)(conv_10)
 
   conv_11 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(pool_4)
   conv_11 = tf.keras.layers.BatchNormalization()(conv_11)
   conv_11 = tf.keras.layers.Activation("relu")(conv_11)
   conv_12 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_11)
   conv_12 = tf.keras.layers.BatchNormalization()(conv_12)
   conv_12 = tf.keras.layers.Activation("relu")(conv_12)
   conv_13 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_12)
   conv_13 = tf.keras.layers.BatchNormalization()(conv_13)
   conv_13 = tf.keras.layers.Activation("relu")(conv_13)
 
   pool_5, mask_5 = MaxPoolWithArgmax2D(pool_size)(conv_13)
   print("Build encoder done..")
 
   # decoder
 
   #print(pool_5.shape)
 
   unpool_1 = MaxUnpool2D(pool_size)([pool_5, mask_5])
 
   conv_14 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(unpool_1)
   conv_14 = tf.keras.layers.BatchNormalization()(conv_14)
   conv_14 = tf.keras.layers.Activation("relu")(conv_14)
   conv_15 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_14)
   conv_15 = tf.keras.layers.BatchNormalization()(conv_15)
   conv_15 = tf.keras.layers.Activation("relu")(conv_15)
   conv_16 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_15)
   conv_16 = tf.keras.layers.BatchNormalization()(conv_16)
   conv_16 = tf.keras.layers.Activation("relu")(conv_16)
 
   unpool_2 = MaxUnpool2D(pool_size)([conv_16, mask_4])
 
   conv_17 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(unpool_2)
   conv_17 = tf.keras.layers.BatchNormalization()(conv_17)
   conv_17 = tf.keras.layers.Activation("relu")(conv_17)
   conv_18 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_17)
   conv_18 = tf.keras.layers.BatchNormalization()(conv_18)
   conv_18 = tf.keras.layers.Activation("relu")(conv_18)
   conv_19 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(conv_18)
   conv_19 = tf.keras.layers.BatchNormalization()(conv_19)
   conv_19 = tf.keras.layers.Activation("relu")(conv_19)
 
   unpool_3 = MaxUnpool2D(pool_size)([conv_19, mask_3])
 
   conv_20 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(unpool_3)
   conv_20 = tf.keras.layers.BatchNormalization()(conv_20)
   conv_20 = tf.keras.layers.Activation("relu")(conv_20)
   conv_21 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(conv_20)
   conv_21 = tf.keras.layers.BatchNormalization()(conv_21)
   conv_21 = tf.keras.layers.Activation("relu")(conv_21)
   conv_22 = tf.keras.layers.Conv2D(128, (kernel, kernel), padding="same")(conv_21)
   conv_22 = tf.keras.layers.BatchNormalization()(conv_22)
   conv_22 = tf.keras.layers.Activation("relu")(conv_22)
 
   unpool_4 = MaxUnpool2D(pool_size)([conv_22, mask_2])
 
   conv_23 = tf.keras.layers.Conv2D(128, (kernel, kernel), padding="same")(unpool_4)
   conv_23 = tf.keras.layers.BatchNormalization()(conv_23)
   conv_23 = tf.keras.layers.Activation("relu")(conv_23)
   conv_24 = tf.keras.layers.Conv2D(64, (kernel, kernel), padding="same")(conv_23)
   conv_24 = tf.keras.layers.BatchNormalization()(conv_24)
   conv_24 = tf.keras.layers.Activation("relu")(conv_24)
 
   unpool_5 = MaxUnpool2D(pool_size)([conv_24, mask_1])
 
   conv_25 = tf.keras.layers.Conv2D(64, (kernel, kernel), padding="same")(unpool_5)
   conv_25 = tf.keras.layers.BatchNormalization()(conv_25)
   conv_25 = tf.keras.layers.Activation("relu")(conv_25)
 
   conv_26 = tf.keras.layers.Conv2D(n_labels, (1, 1), padding="valid")(conv_25)
   conv_26 = tf.keras.layers.BatchNormalization()(conv_26)
   conv_26 = tf.keras.layers.Reshape(
       (input_shape[0] * input_shape[1], n_labels),
       input_shape=(input_shape[0], input_shape[1], n_labels),
   )(conv_26)
 
   outputs = tf.keras.layers.Activation("softmax")(conv_26)
   print("Build decoder done..")
 
   model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SegNet")
 
   return model