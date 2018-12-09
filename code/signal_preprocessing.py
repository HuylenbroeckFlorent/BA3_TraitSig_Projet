import numpy as np

def normalize_monoarray_signal(signal):
	signal=np.array(signal)
	return signal/max(signal)


def split_into_frames(signal, samplerate, frame_size_ms, frame_step_ms):
	signal=np.array(signal)
	frame_size=int(frame_size_ms*samplerate/1000)
	frame_step=int(frame_step_ms*samplerate/1000)

	frames=[]
	for i in range(0,len(signal)-len(signal)%frame_size, frame_step): #misses last (len(signal)%frame_size) values
		frames.append(signal[i:i+frame_size])

	return frames