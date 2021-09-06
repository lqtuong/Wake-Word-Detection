import pyaudio
import wave, pandas, time
import sys, os, csv, getch
import threading
from mplayer import Mplayer

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10

source_dir = "./va_0"

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

class PlayAudio (threading.Thread):
    def __init__(self, wav_filename):
        threading.Thread.__init__(self)
        self.chunk_size = CHUNK
        self.file_name = wav_filename
        try:
            print ('Trying to play file ',self.file_name)
            self.wf = wave.open(self.file_name, 'rb')
        except IOError as ioe:
            sys.stderr.write('IOError on file ' + self.file_name + '\n' + \
            str(ioe) + '. Skipping.\n')
            return
        except EOFError as eofe:
            sys.stderr.write('EOFError on file ' + self.file_name + '\n' + \
            str(eofe) + '. Skipping.\n')
            return
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                                  channels=self.wf.getnchannels(),
                                  rate=self.wf.getframerate(),
                                  output=True)
        self._stop = threading.Event()

    def run(self):
        self._stop.clear()

        while not self._stop.is_set():
            data = self.wf.readframes(self.chunk_size)
            self.stream.write(data)


    def stop(self):
        self._stop.set()
        self.stream.close()
        self.p.terminate()


def insert_label(player):
    all_files = get_filepaths(source_dir)
    with open("label_va.csv","a",newline='') as f:
        writer = csv.writer(f)
        for file in all_files:
            print(file)
            player.set_source(file)
            print(file.split('/')[-1].split('.')[0]," have bird (yes: 1, no: 0): ")
            label = getch.getch()
            print(label)
            writer.writerow([file.split('/')[-1].split('.')[0], str(label)])
            #os.system('mv '+file+' ./00/old')

        print("Thank you")


def threaded_mplayer(player):
    insert_label(player)

def threaded_commandtest(player):
    player.play()


if __name__ == "__main__":
    player = Mplayer()
    mplayer_thread = threading.Thread(target=threaded_mplayer, args=(player,))
    command_thread = threading.Thread(target=threaded_commandtest, args=(player,))

    mplayer_thread.start()
    command_thread.start()

    mplayer_thread.join()
    command_thread.join()







