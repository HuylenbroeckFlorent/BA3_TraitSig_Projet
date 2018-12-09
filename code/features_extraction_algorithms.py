import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft,fft,ifft
from scipy.signal import find_peaks,periodogram
from scipy.io.wavfile import read
from signal_preprocessing import *

def signal_energy(signal):
	signal = np.array(signal)
	return sum(abs(signal)**2)

def mono(signal):
	mono=[]
	for i in signal:
		mono.append(int(sum(i)/len(i)))
	return mono

def autocorrelation_based_pitch_estimation_system(audiopath):
	#0.init
	samplerate, samples_int = read(audiopath)
	samples=np.array(samples_int)

	samples=mono(samples)

	treshold=2.5 #subject to changes
	width=30
	step=10
	#_maxlags=(width*samplerate/1000)-1 #full cover
	_maxlags=(width*samplerate/1000)/2 
	_maxlags=int(_maxlags) 

	#1.normalize the signam
	normalized_samples=normalize_monoarray_signal(samples)

	#2.split the signal into frames
	frames=split_into_frames(normalized_samples, samplerate, width, step)

	#3.compute the energy for each frames
	energy=[]
	for i in range(len(frames)):
		energy.append(signal_energy(frames[i]))

	#4.separate voiced/unvoiced
	to_study=np.array(np.zeros(len(energy)))
	for i in range(len(frames)):
		if energy[i]>treshold:
			to_study[i]=1

	#5.compute the autocorrelation of each frame
	autocorrs=[]
	for i in range(len(frames)):
		if to_study[i]==1:
			a,tmp,b,c=plt.xcorr(frames[i],frames[i],maxlags=_maxlags)
			tmp=tmp[_maxlags:]
			autocorrs.append(tmp)
		else:
			autocorrs.append(np.zeros(_maxlags+1))
	
	#6.Find distances between the two highest peaks for each autocorrelated frames.
	#  First one is always the highest (since we cut the autocorrelation result in half) so we only need 
	#  to find the highest remaining.
	peaks=[]
	for i in range(len(autocorrs)):
		if to_study[i]==1:
			peaks_tmp, peaks_tmp_prop = find_peaks(autocorrs[i], height=0)
			index_max=peaks_tmp[np.argmax(peaks_tmp_prop["peak_heights"])]
			peaks.append(index_max)
		else:
			peaks.append(0)

	#7.compute the fondamental frequencies from the distances between the peaks and the samplerate
	f_zeros=[]
	for i in range(len(peaks)):
		if to_study[i]==1 and peaks[i]!=0:
			f_zeros.append(samplerate/peaks[i])
		else:
			f_zeros.append(0)

	plt.gcf().clear()
	#plt.plot(np.arange(len(peaks))/len(peaks)*(len(samples)/samplerate*1000),peaks,'r')
	plt.plot(np.arange(len(f_zeros))/len(f_zeros)*(len(samples)/samplerate*1000),f_zeros,'b')
	plt.show()

	return f_zeros

def cepstrum_based_pitch_estimation_system(audiopath):
	#0.init
	samplerate, samples_int = read(audiopath)
	samples=np.array(samples_int)

	samples=mono(samples)

	treshold=10.5 #subject to changes
	width=30
	step=10

	#1.normalize the signam
	normalized_samples=normalize_monoarray_signal(samples)

	#2.split the signal into frames
	frames=split_into_frames(normalized_samples, samplerate, width, step)

	#3.compute the energy for each frames
	energy=[]
	for i in range(len(frames)):
		energy.append(signal_energy(frames[i]))

	#4.separate voiced/unvoiced
	to_study=np.array(np.zeros(len(energy)))
	for i in range(len(frames)):
		if energy[i]>treshold:
			to_study[i]=1
	

	#5.for every frame, compute the fft, then log it, and finally compute the ifft of the result
	#  https://en.wikipedia.org/wiki/Cepstrum
	cepstrums = []
	for i in range(len(frames)):
		if to_study[i]==1:
			frame=frames[i]
			fft_frame=fft(frame)
			log_frame=np.log(np.abs(fft_frame))
			ifft_frame=ifft(log_frame)
			cepstrums.append(ifft_frame.real)
			#cepstrums.append(np.abs(ifft_frame))
		else:
			cepstrums.append(0)


	f_zeros_cseptrum=[]
	for i in range(len(cepstrums)):
		if to_study[i]==1:
			cseptrum=cepstrums[i]
		else:
			f_zeros_cseptrum.append(0)

	for i in range(len(cepstrums)):
		if to_study[i]==1:
			plt.plot(cepstrums[i])
			plt.show()
			break

	return cepstrums

def test_both_with_notes():
	#C1 - 65Hz
	#C1s - 69,29Hz
	#D1 - 73,41Hz
	#D1s - 77,78Hz
	#E1 - 82,4Hz
	#F1 - 87,3Hz
	#F1s - 92,49Hz
	#G1 - 97,99Hz
	#G1s - 103,80Hz
	#A1 - 110Hz
	#A1s - 116,54Hz
	#B1 - 123,47Hz
	#C2 - 130,81Hz
	notes = ["C1","C1s","D1","D1s","E1","F1","F1s","G1","G1s","A1","A1s","B1","C2"]
	for note in notes:
		path='./wav-piano-sound-master/wav/'+note+'.wav'
		autocorrelation_based_pitch_estimation_system(path)
		#cepstrum_based_pitch_estimation_system(path)

def test_autocorrelation_based_pitch_estimation_system():
	for i in range(1,2,1):
		path='./cmu_us_bdl_arctic/wav/arctic_a000'+str(i)+'.wav'
		autocorrelation_based_pitch_estimation_system(path)

def test_cepstrum_based_pitch_estimation_system():
	for i in range(1,2,1):
		path='./cmu_us_bdl_arctic/wav/arctic_a000'+str(i)+'.wav'
		cepstrum_based_pitch_estimation_system(path)

test_both_with_notes()