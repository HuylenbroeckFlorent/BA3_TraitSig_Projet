import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
from scipy.fftpack import dct
from scipy.signal import find_peaks,lfilter
from scipy.io.wavfile import read
#import scikits.talkbox as sktb


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


def visualize_study(audiopaths):
	for audiopath in audiopaths:
		print("path : ",audiopath)
		abpes=autocorrelation_based_pitch_estimation_system(audiopath)
		plt.gcf().clear()
		#plt.plot(abpes)
		#plt.show()
		fzero=0
		total=0
		for f0 in abpes:
			if f0!=0 and f0<500:
				fzero=fzero+f0
				total=total+1
		fzero=int(fzero/total)
		print("F0 : ",fzero)

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
		f1=int(f1/total1)
		f2=int(f2/total2)
		f3=int(f3/total3)
		print('formants : ',f1,',',f2,',',f3)

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

def test_mfcc():
	for i in range(1,2,1):
		path='../resources/cmu_us_bdl_arctic/wav/arctic_a000'+str(i)+'.wav'
		a=MFCC(path)
		print(a[0])

#test_mfcc()
#test_with_tones()
#test_autocorrelation_based_pitch_estimation_system()
#test_cepstrum_based_pitch_estimation_system()











path=[]
path.append('../resources/cmu_us_bdl_arctic/wav/arctic_a000'+'6'+'.wav')
path.append('../resources/cmu_us_bdl_arctic/wav/arctic_a000'+'7'+'.wav')
path.append('../resources/cmu_us_bdl_arctic/wav/arctic_a000'+'8'+'.wav')
path.append('../resources/cmu_us_bdl_arctic/wav/arctic_a000'+'9'+'.wav')
path.append('../resources/cmu_us_bdl_arctic/wav/arctic_a00'+'10'+'.wav')

path.append('../resources/cmu_us_slt_arctic/wav/arctic_a000'+'6'+'.wav')
path.append('../resources/cmu_us_slt_arctic/wav/arctic_a000'+'7'+'.wav')
path.append('../resources/cmu_us_slt_arctic/wav/arctic_a000'+'8'+'.wav')
path.append('../resources/cmu_us_slt_arctic/wav/arctic_a000'+'9'+'.wav')
path.append('../resources/cmu_us_slt_arctic/wav/arctic_a00'+'10'+'.wav')
visualize_study(path)


#F0 mec = 120-170 
#F0 meuf = 171 - 300 

#Formants mec F3<1450
#Formants meuf F3>1451