from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from model import CustomSequential

import xml.etree.ElementTree as ET
from pathlib import Path

from preprocess import get_data, image_process

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


###############################################################################################


def run_task(data, epochs=None, batch_size=None):
    """
    Runs model on a given dataset.

    :task: (see preprocess function)
    :param data: Input dataset to train on

    :return trained model
    """
    import model

    ## Retrieve data from tuple
    X0, Y0, X1, Y1, X0_filename, X1_filename = data
    
    X0, X1 = image_process(X0, X1, X0_filename, X1_filename)

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
    args.model.save_weights("dog_weights.hdf5")
    print("save weight done..")

    return args.model


###############################################################################################

if __name__ == "__main__":
    data = get_data()
    run_task(data)
