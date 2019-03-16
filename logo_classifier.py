import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here


def cnn_model_fn(features, labels, mode):
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 50, 50, 3])
    print(input_layer)
    input_layer = tf.cast(input_layer, dtype=tf.float32)
    input_norm = tf.layers.batch_normalization(input_layer)
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_norm,
            filters=16,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu,
            name='conv1')
    print(conv1)

    # Pooling Layer #1
    # conv1_norm = tf.layers.batch_normalization(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
    # pool1 = tf.layers.batch_normalization(pool1)
    # print(pool1)

    # # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)
    print(conv2)
    # conv2 = tf.layers.batch_normalization(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # pool2 = tf.layers.batch_normalization(pool2)
    print(pool2)

    pool2_flat = tf.reshape(pool2, [-1, 9*9*32])
    # dconv1=tf.layers.conv2d_transpose()
    # Dense Layer
    dense1 = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)  # 36/2
    # dropout1 = tf.layers.dropout(
    #     inputs=dense1, rate=0, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dense1, units=24, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=2, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    output = dropout
    logits=output
    predictions = {
        'val': output
    }

    predicted = tf.argmax(input=logits, axis=1)
    lableclass = tf.argmax(input=labels, axis=1)
    print(lableclass.shape)
    print(predicted.shape)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted,
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=output)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        rate = tf.train.exponential_decay(learning_rate=0.1, global_step=tf.train.get_global_step(), decay_steps=100,
                                          decay_rate=0.98, staircase=False)
        train_op = tf.train.AdadeltaOptimizer(learning_rate=rate).minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # eval:
    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=lableclass,
                                       predictions=predicted,
                                       name='acc_op')
        tf.summary.scalar('accuracy', accuracy)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuricy":accuracy}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss,eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    datax = np.load('x.npy')
    datay = np.load('y.npy')
    # datax = datax.reshape([-1, 40, 40, 6])
    # my_feature_columns = [tf.feature_column.numeric_column(key='x', shape=[14400])]
    # train_data = datax[0:340]  # Returns np.array
    # train_labels = datay[0:340]
    #
    # eval_data = datax[300:340]
    # eval_labels = datay[300:340]
    train_data=[]
    train_labels=[]
    eval_data=[]
    eval_labels=[]
    for i in range(0,len(datax)):
        if(i&4==0):
            eval_data.append(datax[i])
            eval_labels.append(datay[i])
        else:
            train_data.append(datax[i])
            train_labels.append(datay[i])
    eval_data=np.array(eval_data)
    eval_labels=np.array(eval_labels)
    train_data=np.array(train_data)
    train_labels=np.array(train_labels)
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir='tensorboard/model'
    )

    # Set up logging for predictions
    tensors_to_log = {"val"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=128,
        num_epochs=None,
        shuffle=True,
        num_threads=4)
    train_res = estimator.train(
        input_fn=train_input_fn,
        steps=10000
    )
    print('train result')
    print(train_res)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=64,
        shuffle=False)
    eval_results = estimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)


tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
    tf.app.run()
