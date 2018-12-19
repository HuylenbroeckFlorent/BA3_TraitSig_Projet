from __future__ import division
import os
import random 
import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
from scipy.fftpack import dct
from scipy.signal import find_peaks,lfilter
from scipy.io.wavfile import read

from lpc import lpc_ref as lpc
from filterbanks import filter_banks
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
			if np.array_equal([],peaks_tmp):
				peaks.append(0)
				to_study[i]=0
			else:
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

def MFCC(audiopath):
	#0.init
	samplerate, samples_int = read(audiopath)
	samples=np.array(samples_int)

	NTFD=512

	#1.split the signal into frames
	width=30
	step=10

	frames=split_into_frames(samples, samplerate, width, step)

	#2.pass the signal into a first order high pass filter
	filtered_frames=[]
	for frame in frames:
		filtered_frames.append(lfilter([1],[1., 0.97],frame))

	#3.apply a hamming window onto the signal
	windowed_frames=[]
	for filtered_frame in filtered_frames:
		windowed_frames.append(filtered_frame*np.hamming(len(filtered_frame)))

	#4.compute the power spectrum
	power_frames=[]
	for windowed_frame in windowed_frames:
		power_frames.append((fft(windowed_frame)**2)/2*NTFD)

	#5.mel-filter
	filter_banks_frames=filter_banks(power_frames,samplerate,nfilt=len(power_frames[0]),NFFT=2*(len(power_frames[0])-1))

	#6.apply direct cosine transform
	_dct=dct(filter_banks_frames,type=2,axis=1,norm='ortho')

	#7.keep the first 13
	_dct=_dct[:13]

	return _dct


def visualize_study(n):
	male_audiopaths=[]
	female_audiopaths=[]
	for i in range(n):
		male_audiopaths.append('../resources/cmu_us_bdl_arctic/wav/'+random.choice(os.listdir('../resources/cmu_us_bdl_arctic/wav/')))
		female_audiopaths.append('../resources/cmu_us_slt_arctic/wav/'+random.choice(os.listdir('../resources/cmu_us_slt_arctic/wav/')))
	
	male_f_zeros=[]
	female_f_zeros=[]

	male_f1=[]
	male_f2=[]
	male_f3=[]

	female_f1=[]
	female_f2=[]
	female_f3=[]

	i=0
	for audiopath in male_audiopaths+female_audiopaths:
		print("processing path : ",audiopath)
		abpes=autocorrelation_based_pitch_estimation_system(audiopath)
		plt.gcf().clear()
		fzero=0
		total=0
		for f0 in abpes:
			if f0!=0 and f0<500:
				fzero=fzero+f0
				total=total+1
		if(i<n):
			male_f_zeros.append(int(fzero/total))
		else:
			female_f_zeros.append(int(fzero/total))

		f=formants(audiopath)
		f1=0
		f2=0
		f3=0
		total1=0
		total2=0
		total3=0
		for form in f:
			if int(form[0])!=0:
				f1=f1+form[0]
				total1=total1+1
			if int(form[1])!=0:
				f2=f2+form[1]
				total2=total2+1
			if int(form[2])!=0:
				f3=f3+form[2]
				total3=total3+1
		if(i<n):
			male_f1.append(int(f1/total1))
			male_f2.append(int(f2/total2))
			male_f3.append(int(f3/total3))
		else:
			female_f1.append(int(f1/total1))
			female_f2.append(int(f2/total2))
			female_f3.append(int(f3/total3))
		i+=1

	print('male f0 : ',male_f_zeros,'; avg = ',int(sum(male_f_zeros)/len(male_f_zeros)))
	print('female f0 : ',female_f_zeros,'; avg = ',int(sum(female_f_zeros)/len(female_f_zeros)))

	print('male F1 : ',male_f1,'; avg = ',int(sum(male_f1)/len(male_f1)))
	print('male F2 : ',male_f2,'; avg = ',int(sum(male_f2)/len(male_f2)))
	print('male F3 : ',male_f3,'; avg = ',int(sum(male_f3)/len(male_f3)))

	print('female F1 : ',female_f1,'; avg = ',int(sum(female_f1)/len(female_f1)))
	print('female F2 : ',female_f2,'; avg = ',int(sum(female_f2)/len(female_f2)))
	print('female F3 : ',female_f3,'; avg = ',int(sum(female_f3)/len(female_f3)))

def rule_based_system(n):
	audiopaths=[]

	for i in range(n):
		if npr.choice([True,False]):
			audiopaths.append('../resources/cmu_us_bdl_arctic/wav/'+random.choice(os.listdir('../resources/cmu_us_bdl_arctic/wav/')))
		else:
			audiopaths.append('../resources/cmu_us_slt_arctic/wav/'+random.choice(os.listdir('../resources/cmu_us_slt_arctic/wav/')))
	
	for audiopath in audiopaths:
		print('processing ',audiopath)
		abpes=autocorrelation_based_pitch_estimation_system(audiopath)
		fzero=0
		total=0
		for f0 in abpes:
			if f0!=0 and f0<500:
				fzero=fzero+f0
				total=total+1
		fzero=fzero/total

		f=formants(audiopath)
		f1=0
		f2=0
		f3=0
		total1=0
		total2=0
		total3=0
		for form in f:
			if int(form[0])!=0:
				f1=f1+form[0]
				total1=total1+1
			if int(form[1])!=0:
				f2=f2+form[1]
				total2=total2+1
			if int(form[2])!=0:
				f3=f3+form[2]
				total3=total3+1

		f1=f1/total1
		f2=f2/total2
		f3=f3/total3

		if fzero<150:
			if f3<1400:
				print("C'est l'homme !")
			else:
				print("C'est certainement l'homme...")

		else:
			if f3>=1400:
				print("C'est la femme !")
			else:
				print("C'est certainement la femme...")

rule_based_system(10)