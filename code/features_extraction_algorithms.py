import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft,fft,ifft
from scipy.signal import find_peaks,lfilter
from scipy.io.wavfile import read
#import scikits.talkbox as sktb
from lpc import lpc_ref as lpc
from signal_preprocessing import *

def signal_energy(signal):
	signal = np.array(signal)
	return sum(abs(signal)**2)

def mono(signal):
	signal=np.array(signal.astype(float))
	if signal.ndim==1:
		return signal
	else:
		return signal.sum(axis=1)/2

def autocorrelation_based_pitch_estimation_system(audiopath):
	#0.init
	samplerate, samples_int = read(audiopath)
	samples=np.array(samples_int)

	samples=mono(samples)

	treshold=2.5 #subject to changes
	width=30
	step=10
	_maxlags=(width*samplerate/1000)-1 #full cover
	#_maxlags=(width*samplerate/1000)/2 
	_maxlags=int(_maxlags) 

	#1.normalize the signal
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

	return f_zeros

def cepstrum_based_pitch_estimation_system(audiopath):
	#0.init
	samplerate, samples_int = read(audiopath)
	samples=np.array(samples_int)

	samples=mono(samples)

	treshold=2.5 #subject to changes
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
			fft_frame=rfft(frame)
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

	return cepstrums

def formants(audiopath):
	#0.init
	samplerate, samples_int = read(audiopath)
	samples=np.array(samples_int)

	#1.split the signal into frames
	width=30
	step=10

	frames=split_into_frames(samples, samplerate, width, step)

	#2.pass the signal into a first order high pass filter
	filtered_frames=[]
	for frame in frames:
		filtered_frames.append(lfilter([1],[1., 0.67],frame))

	#3.apply a hamming window onto the signal
	windowed_frames=[]
	for filtered_frame in filtered_frames:
		windowed_frames.append(filtered_frame*np.hamming(len(filtered_frame)))

	#4.extract the LPC coefficients
	frames_LPC=[]
	ncoeff = int(2 + samplerate / 1000)
	for windowed_frame in windowed_frames:
		a=lpc(windowed_frame, ncoeff)
		frames_LPC.append(a)

	#5.find the roots of the LPC
	frames_roots=[]
	for frame_LPC_coeficient_A in frames_LPC:
		frame_roots=np.roots(frame_LPC_coeficient_A)
		frame_roots=[r for r in frame_roots if np.imag(r) >=0]
		frames_roots.append(frame_roots)

	#6.find the angles
	frames_angles=[]
	for frame_roots in frames_roots:
		frames_angles.append(np.arctan2(np.imag(frame_roots), np.real(frame_roots)))

	#7.deduce the Hz values
	formants=[]
	for frame_angles in frames_angles:
		formants.append(sorted(frame_angles*(samplerate/(2*math.pi))))

	return formants



def test_with_tones():
	tones=["100","1000"]
	#tones=["60","70","80","90","100","300","400","500"]
	#tones=["10","20","30","40","50","60","70","80","90","100","300","400","500","600","700","800","900","1000","1200","1400","1600","1800","2000"]
	file="../resources/sin_waves/"
	wav=".wav"
	for tone in tones:
		path=file+tone+wav
		a=autocorrelation_based_pitch_estimation_system(path)
		#b=cepstrum_based_pitch_estimation_system(path)
		plt.gcf().clear()
		#plt.plot(b)
		plt.plot(a)
		plt.show()
		f=formants(path)
		plt.plot(f)
		plt.show()
		print(f[0])

def test_autocorrelation_based_pitch_estimation_system():
	for i in range(1,2,1):
		path='../resources/cmu_us_bdl_arctic/wav/arctic_a000'+str(i)+'.wav'
		a=autocorrelation_based_pitch_estimation_system(path)
		plt.plot(a)
		plt.show()

def test_cepstrum_based_pitch_estimation_system():
	for i in range(1,2,1):
		path='../resources/cmu_us_bdl_arctic/wav/arctic_a000'+str(i)+'.wav'
		a=cepstrum_based_pitch_estimation_system(path)
		print(sum(a)/len(a))

test_with_tones()
#test_autocorrelation_based_pitch_estimation_system()
#test_cepstrum_based_pitch_estimation_system()