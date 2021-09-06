# python3
from pydub import AudioSegment
import librosa
import librosa.display
import pickle, os, pandas, codecs
import numpy as np
import wave, math

source_dir = '/home/tuong/PycharmProjects/CNNSound/va'
lable_dir = '/home/tuong/PycharmProjects/CNNSound/label_va.csv'

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
    return np.sort(file_paths)

def to1hot(row):
    one_hot = np.zeros(2)
    one_hot[int(row)] = 1.0
    return one_hot

def process_label(directory):
    """
    Data label
    :param directory:
    :return:
    """
    labels = {}
    label = pandas.read_csv(directory, header=None)
    for i in range(len(label[0])):
        labels[str(label[0][i])] = label[1][i]
    return labels

def process_data():
    """
    Process data sound to melspectrogram dataframe
    Save in file .pickle
    :return:
    """
    all_files = get_filepaths('/home/tuong/PycharmProjects/CNNSound/trim')
    all_labels = process_label(lable_dir)

    yy, srr, melss, mfccc, target = [],[],[],[],[]
    count = 1
    for file in all_files:
        print(file)
        y, sr = librosa.core.load(file)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

        log_S = librosa.logamplitude(mels, ref_power=np.max)


        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        label = all_labels[file.split('/')[-1].split('.')[0]]
        yy.append(y)
        srr.append(sr)
        melss.append(log_S)
        mfccc.append(mfcc)
        target.append(label)


        if (count%600 == 0) or (count == len(all_files)) or (count%700==0):
            with open("dataset_va5_"+str(count)+".pickle", 'wb') as output_file:
                df = pandas.DataFrame({'data': yy, 'target': target, 'mels': melss, 'mfcc': mfccc})
                pickle.dump(df, output_file)
            yy, srr, melss, mfccc, target = [],[],[],[],[]
            pass
        count +=1
   

def process():
    """
    Convert audio to spectrogram and save in *.pickle
    :return:
    """
    all_files = get_filepaths(source_dir)
    all_label = process_label(lable_dir)
    yy, srr, melss, mfccc, target = [],[],[],[],[]

    count = 0
    for file in all_files:
        y, sr = librosa.core.load(file)
        i = 0
        label = all_label[file.split('/')[-1].split('.')[0]]
        while (i+3)*sr <= len(y):
            # y[i*sr:(i+2)*sr] is audio file from second i to second i+2
            segment = y[i*sr:(i+3)*sr]
            i += 3
            mels = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
            mfcc = librosa.feature.mfcc(y=segment, sr=sr)
            yy.append(segment)
            srr.append(sr)
            melss.append(mels)
            mfccc.append(mfcc)
            # target.append(label)
            target.append(label)

            if (count%500 == 0) or (count == 3*len(all_files)):
                with open("datasetW"+str(count)+".pickle", 'wb') as output_file:
                    df = pandas.DataFrame({'data': yy, 'target': target, 'mels': melss, 'mfcc': mfccc})
                    pickle.dump(df, output_file)
                yy, srr, melss, mfccc, target = [],[],[],[],[]
                pass
            count +=1

def split_audio(filename, time, label):
    read = wave.open(filename, 'r')
    frameRate = read.getframerate()
    numFrames = read.getnframes()
    duration = numFrames/frameRate
    print(filename.split("/")[-1])
    frames = read.readframes(numFrames)
    x = 0
    while x+time <= duration:
        curFrame = frames[x*frameRate:(x+time+1)*frameRate]
        wf = wave.open(label+'/'+filename.split("/")[-1].split(".")[0]+"_"+str(x)+".wav", 'w')
        wf.setparams(read.getparams())
        wf.setnchannels(read.getnchannels())
        wf.setsampwidth(read.getsampwidth())
        wf.setframerate(frameRate)
        wf.writeframes(curFrame)
        wf.close()
        x += 1

def classify():
    all_files = get_filepaths(source_dir)
    all_label = process_label(lable_dir)
    for file in all_files:
        label = all_label[file.split('/')[-1].split('.')[0]]
        split_audio(file, 3, label)

def trim_audio(filename, start):
    sound = AudioSegment.from_file(filename, format='wav')
    if (start + 0.5)*1000 < len(sound):
        trim = sound[start*1000:(start+0.5)*1000]
        trim.export('/home/tuong/PycharmProjects/CNNSound/trim/'+filename.split("/")[-1].split('.')[0].split('_')[0]+".wav", format='wav')

def trim_process(filepath, startpath):
    all_files = get_filepaths(filepath)
    all_start = pandas.read_csv(startpath, header=None)

    start = {}
    for i in range(len(all_start[0])):
        start[str(all_start[0][i])] = all_start[1][i]

    for file in all_files:
        trim_audio(file, start[file.split('/')[-1]])
def preprocess():
    files = get_filepaths(source_dir)
    for file in files:
        name = file.split("/")[-1].split('_')
        if len(name) == 1:
            print("1")
            trim_audio(file, 0.0)
        else:
            time = file.split("/")[-1].split('-')[0].split('_')[1]
            print("2")
            print(name[1].split('-')[0])
            trim_audio(file, float(time))

if __name__=="__main__":
    process_data()
    # classify()
    ds = pandas.read_pickle("dataset_va5_600.pickle")
    print(ds)
    ds["one_hot_encoding"] = ds.target.apply(to1hot)
    ds["mels_flatten"] = ds.mels.apply(lambda mels: mels.flatten())
    print("shape mels[0]", ds.mels_flatten[2])

    for i in range(20):
       print("shape[0] mels[",i,"]", ds.mels[i].shape, len(ds.mels_flatten[i]))


