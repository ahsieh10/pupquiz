import tensorflow as tf
import numpy as np
from seg_model_layers import segment
import cv2
from PIL import Image as im
from tensorflow.keras.utils import array_to_img
import matplotlib.pyplot as plt

DIMS = (256, 256, 3)
N_LABELS = 3
model = segment(input_shape=DIMS, n_labels=N_LABELS)
model.load_weights("/Users/pranavmahableshwarkar/CS/CSCI1470/segment_weights_35.hdf5")
print(model.summary())

image = "/Users/pranavmahableshwarkar/CS/CSCI1470/pupquiz/segnet/data/images/american_pit_bull_terrier_103.jpg"
arr = cv2.imread(image)
resized = cv2.resize(arr, ((256, 256)))
plt.imshow(resized)
plt.show()
resized = resized.reshape((1, 256, 256, -1))
out = model.predict(resized)
trimap = out.reshape(256, 256, 3)


# mask = model.predict(np.expand_dims(test_image, 0))[0]

def display_mask(pred):
    #plt.imshow(pred)
    mask = np.argmax(pred, axis=-1)
    #mask *= 127
    print(mask.shape)
    plt.imshow(mask)
    plt.show()
    #mask.savefig('temp.png')

display_mask(trimap)

# original = resized.reshape((256, 256, -1))
# segmented = original[trimap != 0] = 0
# cv2.imshow("image", segmented)
# cv2.waitKey()
