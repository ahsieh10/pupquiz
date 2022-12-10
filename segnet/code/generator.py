import cv2
import numpy as np
import tensorflow as tf
 
def category_label(labels, dims, n_labels):
   x = np.zeros([dims[0], dims[1], n_labels])
   for i in range(dims[0]):
       for j in range(dims[1]):
           x[i, j, labels[i][j] - 1] = 1
   x = x.reshape(dims[0] * dims[1], n_labels)
   return x
 
def generator(img_list, mask_list, batch_size, dims, n_labels):
   while True:
       assert len(img_list) == len(mask_list)
       batch = np.random.choice(np.arange(len(img_list)), batch_size)
       imgs = []
       labels = []
       for index in batch:
           original_img = cv2.imread(img_list[index])
           if original_img is None:
               continue
           resized_img = cv2.resize(original_img, dims)
           array_img = tf.keras.utils.img_to_array(resized_img) / 255
           imgs.append(array_img)
 
           original_mask = cv2.imread(mask_list[index])
           resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
           array_mask = category_label(resized_mask[:,:,0], dims, n_labels)
           labels.append(array_mask)
 
       imgs = np.array(imgs)
       labels = np.array(labels)
       yield imgs, labels