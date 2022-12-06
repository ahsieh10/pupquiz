import tensorflow as tf
from model import segment
from generator import generator
import os


def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

fix_gpu()

def main():
    print("beginning training")
    input_dir = "data/images"
    trimap_dir = "data/annotations/trimaps"

    # HYPERPARAMETERS
    BATCH_SIZE = 10
    EPOCHS = 10
    DIMS = (256, 256, 3)
    N_LABELS = 20

    print("reading in files")

    input_image_paths = sorted(
        [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")]
    )

    trimap_image_paths = sorted(
        [os.path.join(trimap_dir,fname) for fname in os.listdir(trimap_dir)
            if fname.endswith(".png") and not fname.startswith(".")]
    )

    print("Splitting validation set.")

    num_val_samples = 1000
    train_input_imgs = input_image_paths[:-num_val_samples]
    train_targets = trimap_image_paths[:-num_val_samples]
    val_input_imgs = input_image_paths[-num_val_samples:]
    val_targets = trimap_image_paths[-num_val_samples:]

    train_generator = generator(
        img_list= train_input_imgs,
        mask_list= train_targets,
        batch_size= BATCH_SIZE,
        dims = [DIMS[0], DIMS[1]],
        n_labels= N_LABELS
    )

    val_generator = generator(
        img_list= val_input_imgs,
        mask_list= val_targets,
        batch_size= BATCH_SIZE,
        dims = [DIMS[0], DIMS[1]],
        n_labels= N_LABELS
    )

    print("creating generators and training the model. ")

    model = segment(input_shape=DIMS, n_labels=N_LABELS)
    print(model.summary())

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=10,
    )

    model.save_weights("segment_weights_" + str(EPOCHS) + ".hdf5")
    print("sava weight done..")

if __name__ == "__main__":
    main()