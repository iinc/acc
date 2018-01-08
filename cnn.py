import tensorflow as tf
import os
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

n_classes = 8
image_width = 51
image_height = 45
channels = 3

batch_size = 128
num_steps = 10000

def convolutional_neural_network(features, labels, mode):
    input_layer = tf.reshape(features, shape=[-1, image_width, image_height, channels])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
        )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
        )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
        )
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
        )
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    flat = tf.reshape(pool4, [-1, pool4.shape[1] * pool4.shape[2] * pool4.shape[3]])
    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=n_classes)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=n_classes)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

model = tf.estimator.Estimator(model_fn=convolutional_neural_network, model_dir="tmp",
            config=tf.contrib.learn.RunConfig(
                save_checkpoints_steps=10,
                save_summary_steps=10,
                save_checkpoints_secs=None,
                num_cores=3,
                session_config=tf.ConfigProto(
                    intra_op_parallelism_threads=3,
                    inter_op_parallelism_threads=3,
                    allow_soft_placement=True,
                    device_count = {'CPU': 3})))

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)


def my_input_fn():
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once('classified/*/*.jpg'), shuffle=True)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    arr = tf.string_split([key], '\\')
    length = tf.cast(arr.dense_shape[1], tf.int32)
    label = arr.values[length - tf.constant(2, dtype=tf.int32)]
    label = tf.string_to_number(label, tf.int32)
    # label = tf.string_split([key], '\\').values[-2]
    image = tf.image.decode_jpeg(value)
    image = tf.cast(image, tf.float32)
    image.set_shape((image_height, image_width, channels))

    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=8*batch_size,
        min_after_dequeue=7*batch_size)

    labels = tf.contrib.slim.one_hot_encoding(labels, n_classes)
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=2)
    images, labels = batch_queue.dequeue()
    return images, labels

model.train(
    input_fn=my_input_fn,
    steps=None,
    hooks=[])
