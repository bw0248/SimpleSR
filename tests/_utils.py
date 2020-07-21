import os
import tensorflow as tf


def load_img_batches(normalize=True):
    img_path = "tests/data/patterns/random_noise"
    img_files = [os.path.join(img_path, fname) for fname in os.listdir(img_path)
                 if os.path.isfile(os.path.join(img_path, fname))]
    batch_1 = _prep_batch(img_files[:len(img_files)//2],  normalize=normalize)
    batch_2 = _prep_batch(img_files[len(img_files)//2:],  normalize=normalize)
    return batch_1, batch_2


def _prep_batch(img_list, normalize):
    batch = tf.data.Dataset.list_files(img_list) \
        .map(tf.io.read_file) \
        .map(tf.image.decode_png)
    if normalize:
        batch = batch.map(lambda img: tf.image.convert_image_dtype(img, dtype=tf.float32))
    return batch.batch(len(img_list))

