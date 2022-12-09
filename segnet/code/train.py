import tensorflow as tf
from model import segment
import numpy as np
import os
import random
 
 
# def fix_gpu():
#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = tf.compat.v1.InteractiveSession(config=config)
 
# fix_gpu()
 
def main():
   print("beginning training")
   input_dir = "/Users/pranavmahableshwarkar/CS/CSCI1470/pupquiz/segnet/data/images"
   trimap_dir = "/Users/pranavmahableshwarkar/CS/CSCI1470/pupquiz/segnet/data/annotations/trimaps"
 
   # HYPERPARAMETERS           
   BATCH_SIZE = 8
   EPOCHS = 10
   DIMS = (256, 256)
   N_LABELS = 3  

   input_image_paths = sorted(
       [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")]
   )
 
   trimap_image_paths = sorted(
       [os.path.join(trimap_dir,fname) for fname in os.listdir(trimap_dir)
           if fname.endswith(".png") and not fname.startswith(".")]
   )
   
   num_imgs = len(input_image_paths)
   random.Random(1470).shuffle(input_image_paths)
   random.Random(1470).shuffle(trimap_image_paths)
   
   def path_to_input_image(path):
        return tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size = DIMS))

   def path_to_target(path):
        img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size = DIMS, color_mode = "grayscale"))
        img = img.astype("uint8") - 1
        return img

   input_imgs = np.zeros((num_imgs,) + DIMS + (3,), dtype = "float32")
   targets = np.zeros((num_imgs,) + DIMS + (1,), dtype = "uint8")
   
   for i in range(num_imgs):
       input_imgs[i] = path_to_input_image(input_image_paths[i])
       targets[i] = path_to_target(trimap_image_paths[i])
       
   num_val_samples = 1000
   train_input_imgs = input_imgs[:-num_val_samples]
   train_targets = targets[:-num_val_samples]
   val_input_imgs = input_imgs[-num_val_samples:]
   val_targets = targets[-num_val_samples:]
   print("creating generators and training the model. ")
   
   model = segment(input_shape=DIMS, n_labels=N_LABELS)
   print(model.summary())
   
   model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

   print(train_input_imgs.shape, train_targets.shape)
   model.fit(train_input_imgs, train_targets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_input_imgs, val_targets)
   )   
   model.save_weights("segment_weights_" + str(EPOCHS) + ".hdf5")
   print("sava weight done..")

if __name__ == "__main__":
   main()
