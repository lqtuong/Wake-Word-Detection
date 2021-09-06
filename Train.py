import tensorflow as tf
from Model import Model
import os, pandas
import numpy as np
from sklearn.metrics import confusion_matrix


source_dir ='/home/tuong/PycharmProjects/CNNSound/'

model = Model()
BATCH_SIZE = model.config.BATCH_SIZE
TRAIN_DATA_NODE =  model.config.train_data_node
TRAIN_DATA_LABEL = model.config.train_labels_node
FC_W = model.config.fc_weights
FC_B = model.config.fc_biases

def get_filepaths(directory):
    """
    Load data file paths
    :param directory:
    :return:
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths

def to1hot(row):
    one_hot = np.zeros(2)
    one_hot[int(row)] = 1.0
    return one_hot

def init(mode):
    """
    Read data and process to train, validation, test
    :return:
    """
    all_files = get_filepaths(source_dir+mode)
    frames =[]
    for file in all_files:
        print("Processing file ",file)
        frames.append(pandas.read_pickle(file))
        pass
    ds = pandas.concat(frames, ignore_index=True)
    ds["one_hot_encoding"] = ds.target.apply(to1hot)
    ds["mels_flatten"] = ds.mels.apply(lambda mels: mels.flatten())
    data_x = np.vstack(ds.mels_flatten).reshape(ds.shape[0],128,22,1).astype(np.float32)
    data_y = np.vstack(ds["one_hot_encoding"])

    return data_x, data_y
def error_rate(predictions, labels):
    #print(predictions)
    correct = np.sum(np.argmax(predictions,1)==np.argmax(labels,1))
    total = predictions.shape[0]
    error = 100.0 - (100*float(correct)/float(total))
    confusions = np.zeros([2, 2], np.float32)
    bundled = zip(np.argmax(predictions,1), np.argmax(labels,1))
    for predicted, actual in bundled:
        confusions[predicted, actual] +=1
    return error, confusions

def trainCRNN():
    # Training computation: logits + cross-entropy loss.
    with tf.name_scope("cost") as scope:
        logits = model.build(TRAIN_DATA_NODE, train=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TRAIN_DATA_LABEL, logits=logits))
        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(FC_W) + tf.nn.l2_loss(FC_B))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers
        tf.summary.scalar("loss", loss)

    # Optimizer: set up a variable that's incremented once per batch and control the learning rate decay
    batch = tf.Variable(0, trainable=False)
    # Decay once per epoch, using an exponential schedule starting at 0.0001.
    with tf.name_scope("learning_rate") as scope:
        learning_rate = tf.train.exponential_decay(0.01, batch * BATCH_SIZE, 500, 0.99, staircase=True)
        # Use simple AdamOptimizer for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
        tf.summary.scalar("learning_rate", learning_rate)

    # constants for validation and tests
    validation_x, validation_y = init("validation0.5")
    test_x, test_y = init("test0.5")
    validation_data_node = tf.constant(validation_x)
    test_data_node = tf.constant(test_x)
    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    validation_prediction = tf.nn.softmax(model.build(validation_data_node))
    test_prediction = tf.nn.softmax(model.build(test_data_node))


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter('log_dir0.5/', sess.graph)
        saver = tf.train.Saver()
        for i in range(2000):
            #Load file to train
            files = get_filepaths("train0.5")
            for file in files:
                #Convert file to data train
                ds = pandas.read_pickle(file)
                ds["one_hot_encoding"] = ds.target.apply(to1hot)
                ds["mels_flatten"] = ds.mels.apply(lambda mels: mels.flatten())
                train_x = np.vstack(ds.mels_flatten).reshape(ds.shape[0],128,22,1).astype(np.float32)
                train_y = np.vstack(ds["one_hot_encoding"])
                train_size = train_y.shape[0]

                for step in range(int(train_size/BATCH_SIZE)):
                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE + 1)
                    batch_data = train_x[offset:(offset+BATCH_SIZE),:,:,:]
                    batch_labels = train_y[offset:(offset+BATCH_SIZE)]
                    feed_dict = {TRAIN_DATA_NODE: batch_data,
                                 TRAIN_DATA_LABEL: batch_labels}
                    summary, _, l, lr, predictions = sess.run([merge, optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
                    writer.add_summary(summary, i)

                    if step%3 == 0:
                        error, confusions_matrix = error_rate(predictions, batch_labels)
                        print("Mini-batch loss: %.5f Error: %.9f Learning rate: %.9f" % (l, error, lr))
                        print(confusion_matrix(np.argmax(predictions,1), np.argmax(batch_labels,1)))
                        print("Validation error: %.5f" % error_rate(validation_prediction.eval(), validation_y)[0])
                        print(confusion_matrix(np.argmax(validation_prediction.eval(),1), np.argmax(validation_y,1)))
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)

        test_error, confusions = error_rate(test_prediction.eval(), test_y)
        # Confusions matrix
        print(confusions)
        print("Test error: %.5f" % test_error)
        print(confusion_matrix(np.argmax(test_prediction.eval(), 1), np.argmax(test_y, 1)))
if __name__=="__main__":
    trainCRNN()