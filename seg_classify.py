import tensorflow as tf
import numpy as np
import cv2
from PIL import Image as im
import matplotlib.pyplot as plt
import segnet.code.model

DIMS = (256, 256, 3)
N_LABELS = 3
model = segnet.code.model.segment(input_shape=DIMS, n_labels=N_LABELS)
model.load_weights("/Users/joedodson/Documents/CS1470/pupquiz/segment_weights_10.hdf5")
print(model.summary())

image = "/Users/joedodson/Documents/cs1470/pupquiz/data/images/american_pit_bull_terrier_103.jpg"
arr = cv2.imread(image)
resized = cv2.resize(arr, ((256, 256)))
plt.imshow(resized)
plt.show()
resized = resized.reshape((1, 256, 256, -1))
out = model.predict(resized)
trimap = out.reshape(256, 256, 3)

# model = keras.models.load_model("oxford_segmentation.keras")
# i = 4
# test_image = val_input_imgs[i]
# plt.axis("off")
# plt.imshow(array_to_img(test_image))

# mask = model.predict(np.expand_dims(test_image, 0))[0]

def display_mask(pred):
    #plt.imshow(pred)
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    print(mask.shape)
    plt.imshow(mask)
    plt.show()
    #mask.savefig('temp.png')

display_mask(trimap)
