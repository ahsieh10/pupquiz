import tensorflow as tf
import tensorflow_datasets as tfds

from model import mnetV2
from preprocessing import pull_processed_data, preprocess

def train(model, data, epochs=25, batch_size=None):
    model.compile(optimizer=tf.keras.optimizers.Adamax(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    train_batches = pull_processed_data(data[0], batch_size)
    test_batches = pull_processed_data(data[1], batch_size)

    history = model.fit(train_batches,
                        epochs=epochs,
                        validation_data=test_batches)
    
    model.save_weights("dog_weights.hdf5")
    print("save weight done..")



if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 10
    DIMS = (256, 256, 3)
    BREEDS = 120

    dataset, info = tfds.load(name="stanford_dogs", with_info=True)
    
    training_data = dataset['train']
    test_data = dataset['test']
    model = mnetV2(DIMS, BREEDS)

    train(model, (training_data, test_data), EPOCHS, BATCH_SIZE)

    
    
