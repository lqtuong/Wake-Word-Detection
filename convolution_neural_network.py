# python3
import tensorflow as tf
import os, pandas, sys, csv
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pyaudio
import wave
import librosa
import librosa.display
from datetime import datetime


CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3

soure_dir = './'
BATCH_SIZE = 50
NUM_CHANNELS = 1
NUM_LABELS = 2
INPUT_SHAPE = (128,130)
SEED = 42

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
    all_files = get_filepaths(soure_dir+mode)
    frames =[]
    for file in all_files:
        print("Processing file ",file)
        frames.append(pandas.read_pickle(file))
        pass
    ds = pandas.concat(frames, ignore_index=True)
    ds["one_hot_encoding"] = ds.target.apply(to1hot)
    ds["mels_flatten"] = ds.mels.apply(lambda mels: mels.flatten())

    data_x = np.vstack(ds.mels_flatten).reshape(ds.shape[0],128,130,1).astype(np.float32)
    data_y = np.vstack(ds["one_hot_encoding"])

    return data_x, data_y

train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1],1))
train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))

# 2x8 convolution, 1 input, 32 output (32 feature maps) 2 8 | 30 8 | 60 10 | 96
conv1_weights = tf.Variable(tf.truncated_normal([2, 8, 1, 32], stddev=0.1, seed=SEED))
conv1_biases = tf.Variable(tf.constant(0.1, shape=[32]))

conv2_weights = tf.Variable(tf.truncated_normal([30, 8, 32, 64], stddev=0.1, seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

fc1_weights = tf.Variable(tf.truncated_normal([1056*64, 512], stddev=0.1, seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.2, shape=[512]))

fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

def model(data, train=False):
    print(data.get_shape())
    # Reshape input to a 2D tensor
    # strides is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 2, 2, 1], padding='SAME')
    print(conv.get_shape())
    # Convolution layer, using our function
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    print("pool_shape: %s" % pool.get_shape())

    conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 2, 2, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    print("pool: %s" % pool.get_shape())
    pool_shape = pool.get_shape().as_list()
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases


def error_rate(predictions, labels):
    #print(predictions)
    correct = np.sum(np.argmax(predictions,1)==np.argmax(labels,1))
    total = predictions.shape[0]
    error = 100.0 - (100*float(correct)/float(total))
    confusions = np.zeros([NUM_LABELS, NUM_LABELS], np.float32)
    bundled = zip(np.argmax(predictions,1), np.argmax(labels,1))
    for predicted, actual in bundled:
        confusions[predicted, actual] +=1
    return error, confusions

def trainCNN():
    with tf.name_scope("cost") as scope:
        logits = model(train_data_node, True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_node, logits=logits))
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        loss += 5e-4 * regularizers
        tf.summary.scalar("loss", loss)

    # Optimizer: set up a variable that's incremented once per batch and control the learning rate decay
    batch = tf.Variable(0)
    with tf.name_scope("learning_rate") as scope:
        learning_rate = tf.maximum(0.000000000001, tf.train.exponential_decay(0.0001, batch * BATCH_SIZE, 500, 0.99, staircase=True))
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
        tf.summary.scalar("learning_rate", learning_rate)

    train_prediction = tf.nn.softmax(logits)
    validation_x, validation_y = init("validation3s")
    test_x, test_y = init("test3s")
    validation_data_node = tf.constant(validation_x)
    test_data_node = tf.constant(test_x)
    validation_prediction = tf.nn.softmax(model(validation_data_node))
    test_prediction = tf.nn.softmax(model(test_data_node))


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter('log_dir3s/', sess.graph)
        saver = tf.train.Saver()
        for i in range(2000):
            files = get_filepaths("train3s")
            for file in files:
                ds = pandas.read_pickle(file)
                ds["one_hot_encoding"] = ds.target.apply(to1hot)
                ds["mels_flatten"] = ds.mels.apply(lambda mels: mels.flatten())
                train_x = np.vstack(ds.mels_flatten).reshape(ds.shape[0],128,130,1).astype(np.float32)
                train_y = np.vstack(ds["one_hot_encoding"])
                train_size = train_y.shape[0]

                for step in range(int(train_size/BATCH_SIZE)):
                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE + 1)
                    batch_data = train_x[offset:(offset+BATCH_SIZE),:,:,:]
                    batch_labels = train_y[offset:(offset+BATCH_SIZE)]
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}
                    summary, _, l, lr, predictions = sess.run([merge, optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
                    writer.add_summary(summary, i)
                    if step%3 == 0:
                        error, confusions_matrix = error_rate(predictions, batch_labels)
                        print("Mini-batch loss: %.5f Error: %.10f Learning rate: %.10f" % (l, error, lr))
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


def play_wav(wav_filename, chunk_size=CHUNK):
    '''
    Play (on the attached system sound device) the WAV file
    named wav_filename.
    '''
    try:
        print ('Trying to play file ',wav_filename)
        wf = wave.open(wav_filename, 'rb')
    except IOError as ioe:
        sys.stderr.write('IOError on file ' + wav_filename + '\n' + \
        str(ioe) + '. Skipping.\n')
        return
    except EOFError as eofe:
        sys.stderr.write('EOFError on file ' + wav_filename + '\n' + \
        str(eofe) + '. Skipping.\n')
        return

    # Instantiate PyAudio.
    p = pyaudio.PyAudio()

    # Open stream.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk_size)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop stream.
    stream.stop_stream()
    stream.close()

    # Close PyAudio.
    p.terminate()

def general_name():
    now = datetime.now()
    return str(now.year) + '-' + \
           str(now.month) + '-' + \
           str(now.day) + '-' + \
           str(now.hour) + ':' + \
           str(now.minute) + ':' + \
           str(now.second) + '.wav'

def prediction(filename):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph('./model.ckpt.meta')
        saver.restore(sess, "./model.ckpt")

        yy, melss, mfccc = [], [], []
        y, sr = librosa.core.load(filename)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

        log_S = librosa.logamplitude(mels, ref_power=np.max)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        yy.append(y)
        melss.append(log_S)
        mfccc.append(mfcc)
        df = pandas.DataFrame({'data': yy,'target': 0, 'mels': melss, 'mfcc': mfccc})
        df["one_hot_encoding"] = df.target.apply(to1hot)
        df["mels_flatten"] = df.mels.apply(lambda mels: mels.flatten())
        data = np.vstack(df.mels_flatten).reshape(df.shape[0], 128, 130, 1).astype(np.float32)
        data_y = np.vstack(df["one_hot_encoding"])

        test_data = tf.constant(data)
        test_prediction = tf.nn.softmax(model(test_data))
        print("output ", np.argmax(test_prediction.eval(), 1))
        #print(conv2_weights.eval(sess))

def record_and_predict():
    while True:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("----------recording----------")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("--------done recording--------")
        stream.stop_stream()
        stream.close()
        p.terminate()

        WAVE_OUTPUT_FILENAME = general_name()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        prediction(WAVE_OUTPUT_FILENAME)

if __name__ == "__main__":
    #trainCNN()
    record_and_predict()



