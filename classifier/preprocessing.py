import tensorflow_datasets as tfds
import tensorflow as tf

DIM = 256
DIMS = (DIM, DIM, 3)
BREEDS = 120

def preprocess(row):
    # Resize the Images and Convert to Floating Type!
    image = tf.image.convert_image_dtype(row['image'], dtype=tf.float32)
    image = tf.image.resize(image, (DIM, DIM), method = 'nearest')

    # Onehot Encoding for the Label!
    label = tf.one_hot(row['label'], BREEDS)

    return image, label

def pull_processed_data(dataset, batch_size=32):
    preproc = dataset.map(preprocess)
    shuffle = preproc.shuffle(buffer_size=1000)
    batched = shuffle.batch(batch_size)
    prep = batched.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return prep

if __name__ == "__main__":
    pull_processed_data()