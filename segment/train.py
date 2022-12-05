import tensorflow as tf
from model import segment
from generator import get_generators
import os

def train(model, train_generator, val_generator, epochs = 50):
    model.compile(optimizer ="rmsprop",
                    loss = "sparse_categorical_crossentropy")

    history = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=len(train_generator),
                                    epochs=epochs,
                                    validation_data=val_generator,
                                    validation_steps=len(val_generator))

    return history

if __name__ == "__main__":
    # Create FCN model
    model = segment(n_labels = 3)

    # The below folders are created using utils.py
    data_dir = "../data/images"
    trimap_dir = "../data/annotations/trimaps"
    
    # If you get out of memory error try reducing the batch size
    BATCH_SIZE=8

    train_generator, val_generator = get_generators(data_dir, trimap_dir, BATCH_SIZE)

    EPOCHS=50
    history = train(model, train_generator, val_generator, epochs=EPOCHS)