import os, torch, numpy, cv2, imageio, random, python_speech_features
import matplotlib.pyplot as plt
from scipy.io import wavfile
from glob import glob
from torchvision.transforms import RandomCrop
from scipy import signal

def get_noise_list(musanPath, rirPath):
	augment_files = glob(os.path.join(musanPath, '*/*/*/*.wav'))
	noiselist = {}
	rir = numpy.load(rirPath)
	for file in augment_files:
		if not file.split('/')[-4] in noiselist:
			noiselist[file.split('/')[-4]] = []
		noiselist[file.split('/')[-4]].append(file)
	return rir, noiselist

def augment_wav(audio, aug_type, rir, noiselist):
	if aug_type == 'rir':
		rir_gains = numpy.random.uniform(-7,3,1)
		rir_filts = random.choice(rir)
		rir     = numpy.multiply(rir_filts, pow(10, 0.1 * rir_gains))    
		audio   = signal.convolve(audio, rir, mode='full')[:len(audio)]
	else:
		noisecat = aug_type
		noisefile = random.choice(noiselist[noisecat].copy())
		snr = [random.uniform({'noise':[0,15],'music':[5,15]}[noisecat][0], {'noise':[0,15],'music':[5,15]}[noisecat][1])]
		_, noiseaudio = wavfile.read(noisefile)
		if len(noiseaudio) < len(audio):
			shortage = len(audio) - len(noiseaudio)
			noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
		else:
			noiseaudio = noiseaudio[:len(audio)]

		noise_db = 10 * numpy.log10(numpy.mean(abs(noiseaudio ** 2)) + 1e-4)
		clean_db = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
		noise = numpy.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noiseaudio
		audio = audio + noise
	return audio.astype(numpy.int16)

def load_audio(data, data_path, length, start, end, audio_aug, rirlist = None, noiselist = None):
	# Find the path of the audio data
	data_type = data[0]
	id_name = data[1][:8]
	file_name = data[1].split('/')[0] + '_' + data[1].split('/')[1] + '_' + data[1].split('/')[2] + \
	'_' + data[2].split('/')[0] + '_' + data[2].split('/')[1] + '_' + data[2].split('/')[2] + '.wav'
	audio_file_path = os.path.join(data_path, data_type, id_name, file_name)
	# Load audio, compute MFCC, cut it to the required length
	_, audio = wavfile.read(audio_file_path)

	if audio_aug == True:
		augtype = random.randint(0,3)
		if augtype == 1: # rir
			audio = augment_wav(audio, 'rir', rirlist, noiselist)
		elif augtype == 2:
			audio = augment_wav(audio, 'noise', rirlist, noiselist)   
		elif augtype == 3:
			audio = augment_wav(audio, 'music', rirlist, noiselist)
		else:
			audio = audio

	feature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
	length_audio = int(round(length * 100))
	if feature.shape[0] < length_audio:
		shortage    = length_audio - feature.shape[0]
		feature     = numpy.pad(feature, ((0, shortage), (0,0)), 'wrap')
	feature = feature[int(round(start * 100)):int(round(end * 100)),:]
	return feature

def load_video(data, data_path, length, start, end, visual_aug):	
	# Find the path of the visual data
	data_type = data[0]
	id_name = data[1][:8]
	file_name = data[1].split('/')[0] + '_' + data[1].split('/')[1] + '_' + data[1].split('/')[2] + \
	'_' + data[2].split('/')[0] + '_' + data[2].split('/')[1] + '_' + data[2].split('/')[2] + '.mp4'
	video_file_path = os.path.join(data_path, data_type, id_name, file_name)
	# Load visual frame-by-frame, cut it to the required length
	length_video = int(round((end - start) * 25))
	video = cv2.VideoCapture(video_file_path)
	faces = []
	augtype = 'orig'

	if visual_aug == True:
		new = int(112*random.uniform(0.7, 1))
		x, y = numpy.random.randint(0, 112 - new), numpy.random.randint(0, 112 - new)
		M = cv2.getRotationMatrix2D((112/2,112/2), random.uniform(-15, 15), 1)
		augtype = random.choice(['orig', 'flip', 'crop', 'rotate'])

	num_frame = 0
	while video.isOpened():
		ret, frames = video.read()
		if ret == True:
			num_frame += 1
			if num_frame >= int(round(start * 25)) and num_frame < int(round(end * 25)):
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				if augtype == 'orig':
					faces.append(face)
				elif augtype == 'flip':
					faces.append(cv2.flip(face, 1))
				elif augtype == 'crop':
					faces.append(cv2.resize(face[y:y+new, x:x+new] , (112,112))) 
				elif augtype == 'rotate':
					faces.append(cv2.warpAffine(face, M, (112,112)))
		else:
			break
	video.release()
	faces = numpy.array(faces)
	if faces.shape[0] < length_video:
		shortage    = length_video - faces.shape[0]
		faces     = numpy.pad(faces, ((0,shortage), (0,0),(0,0)), 'wrap')	
	# faces = numpy.array(faces)[int(round(start * 25)):int(round(end * 25)),:,:]
	return faces

def load_label(data, length, start, end):
	labels_all = []
	labels = []
	data_type = data[0]
	start_T, end_T, start_F, end_F = float(data[4]), float(data[5]), float(data[6]), float(data[7])	
	for i in range(int(round(length * 100))):
		if data_type == 'TAudio':
			labels_all.append(1)
		elif data_type == 'FAudio' or data_type == 'FSilence':
			labels_all.append(0)
		else:
			if i >= int(round(start_T * 100)) and i <= int(round(end_T * 100)):
				labels_all.append(1)
			else:
				labels_all.append(0)
	for i in range(int(round(length * 25))):
		labels.append(int(round(sum(labels_all[i*4: (i+1)*4]) / 4)))
	return labels[round(start*25): round(end*25)]

class loader_TalkSet(object):
	def __init__(self, trial_file_name, data_path, audio_aug, visual_aug, musanPath, rirPath,**kwargs):
		self.data_path = data_path
		self.audio_aug = audio_aug
		self.visual_aug = visual_aug
		self.minibatch = []
		self.rir, self.noiselist = get_noise_list(musanPath, rirPath)
		mix_lst = open(trial_file_name).read().splitlines()
		mix_lst = list(filter(lambda x: float(x.split()[3]) >= 1, mix_lst)) # filter the video less than 1s
		# mix_lst = list(filter(lambda x: x.split()[0] == 'TSilence', mix_lst))
		sorted_mix_lst = sorted(mix_lst, key=lambda data: (float(data.split()[3]), int(data.split()[-1])), reverse=True)		
		start = 0
		while True:
			length_total = float(sorted_mix_lst[start].split()[3])
			batch_size = int(250 / length_total)
			end = min(len(sorted_mix_lst), start + batch_size)
			self.minibatch.append(sorted_mix_lst[start:end])
			if end == len(sorted_mix_lst):
				break
			start = end
		# self.minibatch = self.minibatch[0:5]

	def __getitem__(self, index):
		batch_lst = self.minibatch[index]
		length_total = float(batch_lst[-1].split()[3])
		length_total = (int(round(length_total * 100)) - int(round(length_total * 100)) % 4) / 100
		audio_feature, video_feature, labels = [], [], []
		duration = random.choice([1,2,4,6])
		#duration = 6
		length = min(length_total, duration)
		if length == duration:
			start = int(round(random.randint(0, round(length_total * 25) - round(length * 25)) * 0.04 * 100)) / 100
			end = int(round((start + length) * 100)) / 100
		else:
			start, end = 0, length

		for line in batch_lst:
			data = line.split()
			audio_feature.append(load_audio(data, self.data_path, length_total, start, end, audio_aug = self.audio_aug, rirlist = self.rir, noiselist = self.noiselist))
			video_feature.append(load_video(data, self.data_path, length_total, start, end, visual_aug = self.visual_aug))        
			labels.append(load_label(data, length_total, start, end))

		return torch.FloatTensor(numpy.array(audio_feature)), \
			   torch.FloatTensor(numpy.array(video_feature)), \
			   torch.LongTensor(numpy.array(labels))

	def __len__(self):
		return len(self.minibatch)