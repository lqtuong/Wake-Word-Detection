#import tensorflow as tf
import pyaudio
import wave, librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys, csv, os
from datetime import datetime

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


def join_audio(audios):
    data = []
    for infile in audios:
        w = wave.open(infile,'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    WAVE_OUTPUT_FILENAME = general_name()
    output = wave.open('join_'+WAVE_OUTPUT_FILENAME,'wb')
    output.setparams(data[0][0])
    for frame in range(len(data)):
        output.writeframes(data[frame][1])
    output.close()

def sound_plot(filename):
    # Load sound file
    y, sr = librosa.load(filename)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)

    # Make a new figure
    plt.figure(figsize=(12, 4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    #librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename.split('.')[0]+".png")


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




if __name__=="__main__":
    #sound_plot('4.wav')
    record_sound(1)
    # files = get_filepaths("/home/tuong/PycharmProjects/CNNSound/va")
    # for file in files:
    #     sound_plot(file)



