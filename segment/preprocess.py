import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import random
import model

input_dir = "data/images"
trimap_dir = "data/annotations/trimaps"

input_image_paths = sorted(
    [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")]
)

trimap_image_paths = sorted(
    [os.path.join(trimap_dir,fname) for fname in os.listdir(trimap_dir)
        if fname.endswith(".png") and not fname.startswith(".")]
)

img_size = (256,256)
num_imgs = len(input_image_paths)

random.Random(1337).shuffle(input_image_paths)
random.Random(1337).shuffle(trimap_image_paths)

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size = img_size))

def path_to_target(path):
    img = img_to_array(load_img(path, target_size = img_size, color_mode = "grayscale"))
    img = img.astype("uint8") - 1
    return img

input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype = "float32")

targets = np.zeros((num_imgs,) + img_size + (1,), dtype = "uint8")

for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_image_paths[i])
    targets[i] = path_to_target(trimap_image_paths[i])

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]

seg = model.segment((256,256,3), 3)

seg.compile(optimizer ="rmsprop", loss = "sparse_categorical_crossentropy")

seg.summary()

'''
seg.fit(train_input_imgs, train_targets,
                    epochs=50,
                    batch_size=64,
                    validation_data=(val_input_imgs, val_targets))
'''