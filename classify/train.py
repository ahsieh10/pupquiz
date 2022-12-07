from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from model import CustomSequential

import xml.etree.ElementTree as ET
from pathlib import Path

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


###############################################################################################

def read_content(xml_file: str):
    '''taken and modified from https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python'''

    tree = ET.parse(xml_file)
    root = tree.getroot()

    box = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        box = [xmin, ymin, xmax, ymax]

    return box


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
        "oxford_iiit_pet", split=["train[:80%]", "train[80%:]"])
    
    X0, X1 = [[r['image'] for r in tfds.as_numpy(D)] for D in (D0, D1)]
    X0_filename, X1_filename = [[r['file_name'].decode('ascii') for r in tfds.as_numpy(D)] for D in (D0, D1)]
    Y0, Y1 = [np.array([r['label'] for r in tfds.as_numpy(D)]) for D in (D0, D1)]


    return X0, Y0, X1, Y1, X0_filename, X1_filename


###############################################################################################

def preprocess(task, X0, X1, X0_filename, X1_filename):
    """
    Crops images down to bounding box and resizes them to 150 by 150 tensors.

    :param task: 
    :param X0: Training image set
    :param X1: Test image set
    :param X0_filename: Training image filenames
    :param X1_filename: Test image filenames

    """
    input_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255),
            tf.keras.layers.Resizing(170, 170),
        ]
    )
    for i in range(len(X0)):
        if task == 1:
            file_path = "annotations/xmls/" + X0_filename[i][0: X0_filename[i].find('.jpg')] + ".xml"
            one_file = Path(file_path)
            if one_file.exists():
                box = read_content(file_path)
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                cropped = X0[i][ymin:ymax, xmin:xmax, :]
        else:
            cropped = X0[i]
        X0[i] = input_prep_fn(tf.convert_to_tensor(cropped))

    for i in range(len(X1)):
        if task == 1:
            file_path = "annotations/xmls/" + X1_filename[i][0: X1_filename[i].find('.jpg')] + ".xml"
            one_file = Path(file_path)
            print(file_path)
            if one_file.exists():
                box = read_content(file_path)
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                cropped = X1[i][ymin:ymax, xmin:xmax, :]
        else:
            cropped = X1[i]
        X1[i] = input_prep_fn(tf.convert_to_tensor(cropped))

    X0 = tf.convert_to_tensor(X0)
    X1 = tf.convert_to_tensor(X1)
    
    return X0, X1


def run_task(task, data, epochs=None, batch_size=None):
    """
    Runs model on a given dataset.

    :task: (see preprocess function)
    :param data: Input dataset to train on

    :return trained model
    """
    import model

    ## Retrieve data from tuple
    X0, Y0, X1, Y1, X0_filename, X1_filename = data
    
    X0, X1 = preprocess(task, X0, X1, X0_filename, X1_filename)

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

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--task",    default=2,     choices='1 2'.split(), help="task to perform")
    args = parser.parse_args()

    data = get_data()
    run_task(args.task, data)
