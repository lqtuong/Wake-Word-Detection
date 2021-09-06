import numpy as np
import tensorflow as tf
import librosa, pandas, time, os
from Model import Model
import pyaudio, csv, wave
from datetime import datetime
#import RPi.GPIO as GPIO
model = Model()
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3


def general_name():
    now = datetime.now()
    return str(now.year) + '-' + \
           str(now.month) + '-' + \
           str(now.day) + '-' + \
           str(now.hour) + ':' + \
           str(now.minute) + ':' + \
           str(now.second) + '.wav'
def record_sound(label):
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
    print ("-------done recording-------")
    stream.stop_stream()
    stream.close()
    p.terminate()
    with open("label"+str(label)+".csv", "a", newline='') as f:
        writer = csv.writer(f)
        WAVE_OUTPUT_FILENAME = general_name()
        writer.writerow([WAVE_OUTPUT_FILENAME, str(label)])
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    sound_plot(WAVE_OUTPUT_FILENAME)

def prediction(file):
   #record()
   with tf.Session() as sess:
       saver = tf.train.Saver()
       saver.restore(sess, "./model.ckpt")
       print("Start")
       #os.system("arecord -D plughw:1,0 -d 3 in.wav > /dev/null 2>&1")
       start_time = time.time()
       y, sr = librosa.core.load(file)
       mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
       log_s = librosa.logamplitude(mels, ref_power=np.max)
       mfcc = librosa.feature.mfcc(y=y, sr=sr)

       df = pandas.DataFrame({'data': list([y]), 'mels': list([log_s]), 'mfcc': list([mfcc])})
       df["mels_flatten"] = df.mels.apply(lambda mels: mels.flatten())
       data = np.vstack(df.mels_flatten).reshape(df.shape[0], 128, 22, 1).astype(np.float32)
       #test_data = tf.constant(data)

       #test_prediction = tf.nn.softmax(model.build(model.config.test_data_node))
       # with tf.Session() as sess:
       #     saver = tf.train.Saver()
       #     saver.restore(sess, "./model.ckpt")
       #     print(test_prediction.eval())
       #     print("output ",np.argmax(test_prediction.eval(),1))

       #test_prediction = tf.nn.softmax(model.build(model.config.train_data_node))

           # print(np.argmax([[ 0.46238881, 0.53761113]],1))

       pr = sess.run(model.test_predict , feed_dict={model.config.test_data_node: data})

       out = pr.argmax(axis=1)[0]
       print(out)
       time_sleep = 0.0

       # if out == 1:
       #     GPIO.setmode(GPIO.BCM)
       #     GPIO.setwarnings(False)
       #     GPIO.setup(18, GPIO.OUT)
       #     print
       #     "LED on"
       #     GPIO.output(18, GPIO.HIGH)
       #     time.sleep(0.5)
       #     print
       #     "LED off"
       #     GPIO.output(18, GPIO.LOW)
       #     time_sleep = 0.5
       # print('time', time.time() - start_time - time_sleep)

#prediction()
