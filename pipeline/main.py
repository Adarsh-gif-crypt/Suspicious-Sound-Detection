import numpy as np
import pyaudio
import librosa
import noisereduce as nr
from keras.models import load_model

RATE = 22050
CHUNK_SIZE = 22050

model = load_model('../model/model.h5')

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)
audio_buffer = np.zeros((0,))

data = stream.read(5000)
noise_sample = np.frombuffer(data, dtype=np.float32)
loud_threshold = np.mean(np.abs(noise_sample)) * 10


while True:

    data = stream.read(CHUNK_SIZE)
    audio_chunk = np.frombuffer(data, dtype=np.float32)
    current_window = nr.reduce_noise(y=audio_chunk, y_noise=noise_sample, sr = RATE)
    
    audio_buffer = np.concatenate((audio_buffer, current_window))

    if audio_buffer.size >= RATE*3:

        audio_clip = audio_buffer[-RATE*3:]

        #if (np.mean(np.abs(audio_clip))<loud_threshold):
        #    print('Inside Silence Threshold')
        #    audio_buffer = np.zeros((0,))
        #else:    
        mfcc = librosa.feature.mfcc(y=audio_clip,sr=RATE,n_mfcc = 50)
        scaled_mfcc = np.mean(mfcc.T,axis = 0)
        scaled_mfcc_reshaped = scaled_mfcc.reshape((1, 50, 1))
        #print(scaled_mfcc_reshaped)
        pred = model.predict(scaled_mfcc_reshaped)
        if pred[0][np.argmax(pred)]>0.5 and np.argmax(pred)==2:
            print('Gun Shot Detected')
        audio_buffer = np.zeros((0,))