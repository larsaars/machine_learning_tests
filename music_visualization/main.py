#!/usr/bin/env python3

'''
visualize mic input audio as 2d image
with hilbert curve
'''

import numpy as np
import pyaudio
import time
import librosa
import matplotlib.pyplot as plt
from matplotlib import animation
from multiprocessing import Process, Manager 


class AudioHandler(object):
    def __init__(self, feature_callback):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None
        self.feature_callback = feature_callback

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        self.feature_callback(librosa.feature.mfcc(numpy_array))
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): 
            time.sleep(2.0)



def plotting(data):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    im = ax.imshow(data.value, animated=True)

    def update_image(i):
        im.set_array(data.value)
        time.sleep(np.random.uniform(0.1, 0.3))
        # plt.pause(0.5)
    ani = animation.FuncAnimation(fig, update_image, interval=0)

    plt.show()


def listening(data):
    try:
        def feature_callback(x):
            time.sleep(np.random.uniform(0.1, 0.3))
            data.value = x
        audio = AudioHandler(feature_callback)
        audio.start()     
        audio.mainloop()
    except KeyboardInterrupt:
        audio.stop()




if __name__ == '__main__':
    manager = Manager()
    data = np.random.rand(20, 5)
    data = manager.Value('data', data)
    Process(target=plotting, args=(data,)).start()
    Process(target=listening, args=(data,)).start()



