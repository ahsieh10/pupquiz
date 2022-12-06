from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from model import CustomSequential

import xml.etree.ElementTree as ET

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


###############################################################################################

def read_content(xml_file: str):
    '''taken from https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python'''

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


def get_data():
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

    D0, D1 = tfds.load(
            "oxford_iiit_pet", split=["train[:80%]", "test"])
    
    X0, X1 = [[r['image'] for r in tfds.as_numpy(D)] for D in (D0, D1)]
    Y0, Y1 = [np.array([r['label'] for r in tfds.as_numpy(D)]) for D in (D0, D1)]


    return X0, Y0, X1, Y1, D0, D1


###############################################################################################

def preprocess(X0, X1):
    input_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255),
            tf.keras.layers.Resizing(150, 150),
        ]
    )

    for i in range(len(X0)):
        X0[i] = input_prep_fn(tf.convert_to_tensor(X0[i]))
    
    for i in range(len(X1)):
        X1[i] = input_prep_fn(tf.convert_to_tensor(X1[i]))

    X0 = tf.convert_to_tensor(X0)
    X1 = tf.convert_to_tensor(X1)
    
    return X0, X1


def run_task(data, epochs=None, batch_size=None):
    """
    Runs model on a given dataset.

    :param data: Input dataset to train on

    :return trained model
    """
    import model

    ## Retrieve data from tuple
    X0, Y0, X1, Y1, D0, D1 = data
    
    X0, X1 = preprocess(X0, X1)

    args = model.get_default_CNN_model()

    ## Prioritize function arguments
    if epochs is None:
        epochs = args.epochs
    if batch_size is None:
        batch_size = args.batch_size

    # Training model
    print("Starting Model Training")
    history = args.model.fit(
        X0, Y0,
        epochs          = epochs,
        batch_size      = batch_size,
        validation_data = (X1, Y1),
    )

    return args.model


###############################################################################################

if __name__ == "__main__":
    import argparse

    data = get_data()
    run_task(data)
