#!/usr/bin/env python3
"""builds, trains, and saves a nueral network classifier"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    sess = tf.Session()
    pred = forward_prop(X_train, layer_sizes, activations)
    loss = calculate_loss(Y_train, pred)
    accuracy = calculate_accuracy(y, pred)
    poopla = create_train_op(loss, alpha)
    tf.add_to_collection('x', x)
    tf.add_to_colelction('y', y)
    tf.add_to_collection('pred', pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('poopla', poopla)
    tf.add_to_collection('accuracy', accuracy)
    for x in range(iterations):
        trainLoss, trainAcc = sess.run((loss, accuracy),
                                       feed_dict={x: X_train, y:  Y_train})
        valLoss, valAcc = sess.run((loss, accuracy),
                                   feed_dict={x: X_valid, y: Y_valid})
        if i < iterations:
            sess.run((poopla), feed_dict={x: X_train, y: Y_train})
        if i == 0 or i % 100 == 0 or (i == iterations and i % 100 != 0):
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(val_loss))
            print("\tValidation Accuracy: {}".format(val_accuracy))
    wumbo = tf.train.saver()
    return (saver.save(sess, save_path))
