import os
import numpy as np
import cv2
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

class Generator(tf.keras.utils.Sequence):
    def __init__(self, data, trimap, batch_size=32, shuffle_images=True, image_min_side=24):
        self.batch_size = batch_size
        self.shuffle_images = shuffle_images
        self.image_min_side = image_min_side
        self.input_image_paths = data
        self.trimap_image_paths = trimap
        self.create_image_groups()

    def create_image_groups(self):
        # Create groups of BATCH_SIZE for generator
        self.image_groups = [[self.input_image_paths[x % len(self.input_image_paths)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.input_image_paths), self.batch_size)]
        self.trimap_groups = [[self.trimap_image_paths[x % len(self.trimap_image_paths)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.trimap_image_paths), self.batch_size)]

    def resize_image(self, img, min_side_len):
        h, w, c = img.shape

        # limit the min side maintaining the aspect ratio
        if min(h, w) < min_side_len:
            im_scale = float(min_side_len) / h if h < w else float(min_side_len) / w
        else:
            im_scale = 1.

        new_h = int(h * im_scale)
        new_w = int(w * im_scale)

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return re_im, new_h / h, new_w / w

    def load_images(self, image_group):
        images = []
        for img in image_group:
            img, rh, rw = self.resize_image(img)
            images.append(img)

        return images
    
    def construct_image_batch(self, image_group):
         # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype='float32')

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating the batches.
        """
        image_group = self.image_groups[index]
        trimap_group = self.trimap_groups[index]
        images = self.load_images(image_group)
        image_batch = self.construct_image_batch(images)

        return np.array(image_batch), np.array(trimap_group)


def resize_image(self, img, min_side_len):
    h, w, c = img.shape

    # limit the min side maintaining the aspect ratio
    if min(h, w) < min_side_len:
        im_scale = float(min_side_len) / h if h < w else float(min_side_len) / w
    else:
        im_scale = 1.

    new_h = int(h * im_scale)
    new_w = int(w * im_scale)

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return re_im #, new_h / h, new_w / w

def load_images(self, image_group):
    images = []
    for image_path in image_group:
        img = img_to_array(load_img(image_path))
        img = self.resize_image(img)
        images.append(img)

    return images


def get_generators(data_dir, trimap_dir, BATCH_SIZE):
    # data_dir = "../data/images"
    # trimap_dir = "../data/annotations/trimaps"
    input_image_paths = sorted(
        [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(".jpg")]
    )
    trimap_image_paths = sorted(
        [os.path.join(trimap_dir,fname) for fname in os.listdir(trimap_dir)
            if fname.endswith(".png") and not fname.startswith(".")]
    )
    assert len(input_image_paths) == len(trimap_image_paths)

    num_imgs = len(input_image_paths)

    seed = 1470
    np.random.seed(seed)
    np.random.shuffle(input_image_paths)
    np.random.seed(seed)
    np.random.shuffle(trimap_image_paths)

    def path_to_input_image(path):
        return img_to_array(load_img(path))

    def path_to_target(path):
        img = img_to_array(load_img(path, color_mode = "grayscale"))
        img = img.astype("uint8") - 1
        return img

    input_imgs = []
    targets = []

    for i in range(num_imgs):
        input_imgs.append(path_to_input_image(input_image_paths[i]))
        targets.append(path_to_target(trimap_image_paths[i]))

    num_val_samples = 1000
    train_input_imgs = input_imgs[:-num_val_samples]
    train_targets = targets[:-num_val_samples]
    val_input_imgs = input_imgs[-num_val_samples:]
    val_targets = targets[-num_val_samples:]

    train_generator = Generator(train_input_imgs, train_targets)
    val_generator = Generator(val_input_imgs, val_targets)
    return train_generator, val_generator

    