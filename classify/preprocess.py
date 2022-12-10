import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

def remove_background(data, filename):
    file_path = "data/trimaps/" + filename[0: filename.find('.jpg')] + ".png"
    image = Image.open(file_path) #TRIMap
    trimap = cv2.imread(file_path)
    arr_trimap = cv2.resize(trimap, (data.shape[1], data.shape[0]))

    data[arr_trimap == 2] = 0
    data[arr_trimap == 3] = 0
    return data

def get_data(dataset):
    """
    Loads "oxford_iiit_pet" training and testing datasets

    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
            D0: TF Dataset training subset
            D1: TF Dataset testing subset
    """

    import tensorflow_datasets as tfds

    if dataset == "stanford_dogs":
        train_string = "[:20%]"
    elif dataset == "oxford_iiit_pet":
        train_string = ""
    D0, D1 = tfds.load(
        dataset, split=["train" + train_string, "test" + train_string])

    if dataset == "stanford_dogs":
        X0_filename, X1_filename = [[r['image/filename'].decode('ascii') for r in tfds.as_numpy(D)] for D in (D0, D1)]
    elif dataset == "oxford_iiit_pet":
        X0_filename, X1_filename = [[r['file_name'].decode('ascii') for r in tfds.as_numpy(D)] for D in (D0, D1)]
  
    X0, X1 = [[r['image'] for r in tfds.as_numpy(D)] for D in (D0, D1)]
    Y0, Y1 = [np.array([r['label'] for r in tfds.as_numpy(D)]) for D in (D0, D1)]


    return X0, Y0, X1, Y1, X0_filename, X1_filename

def image_process(X0, X1, X0_filename, X1_filename, dataset):
    """
    Image preprocessing (image resizing, run validation set through background removal function)

    :param X0: Training image set
    :param X1: Test image set
    :param X0_filename: Training image filenames
    :param X1_filename: Test image filenames

    """
    input_prep_fn = tf.keras.Sequential(
        [tf.keras.layers.Rescaling(scale=1 / 255),
        tf.keras.layers.Resizing(256, 256)])

    for i in range(len(X0)):
        if dataset == "oxford_iiit_pet":
            X0[i] = remove_background(X0[i], X0_filename[i])
        X0[i] = input_prep_fn(tf.convert_to_tensor(X0[i]))

    for i in range(len(X1)):
        if dataset == "oxford_iiit_pet":
            X1[i] = remove_background(X1[i], X1_filename[i])
        X1[i] = input_prep_fn(tf.convert_to_tensor(X1[i]))
    X0 = tf.convert_to_tensor(X0)
    X1 = tf.convert_to_tensor(X1)
    
    return X0, X1


# if __name__ == "__main__":
#     import tensorflow_datasets as tfds

#     D0, D1 = tfds.load(
#         "oxford_iiit_pet", split=["train[:80%]", "train[80%:]"])
    
#     X0, X1 = [[r['image'] for r in tfds.as_numpy(D)] for D in (D0, D1)]
#     X0_filename, X1_filename = [[r['file_name'].decode('ascii') for r in tfds.as_numpy(D)] for D in (D0, D1)]
#     remove_background(X0, X0_filename)