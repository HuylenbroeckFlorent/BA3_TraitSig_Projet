import numpy as np
from scipy.io import wavfile

def generate_wav_files(path,f,Fs,duration,volume):
	wavfile.write(path+str(int(f))+'.wav', Fs, volume*np.sin(2*np.pi*np.arange(Fs*duration)*f/Fs))

def generate():
	frequencies=[10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
	sample_rate=44100
	duration=2.0 #duration in s
	volume=1.0 #volume between 0.0 and 1.0
	path='../resources/sin_waves/'

	for i in frequencies:
		generate_wav_files(path,i,sample_rate,duration,volume)

generate()