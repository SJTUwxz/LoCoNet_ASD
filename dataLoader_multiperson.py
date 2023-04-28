import os, torch, numpy, cv2, random, glob, python_speech_features, json, math
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
from operator import itemgetter
from torchvggish import vggish_input, vggish_params, mel_features


def overlap(audio, noiseAudio):
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio**2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio**2)) + 1e-4)
    noiseAudio = numpy.sqrt(10**((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio
    return audio.astype(numpy.int16)


def load_audio(data, dataPath, numFrames, audioAug, audioSet=None):
    dataName = data[0]
    fps = float(data[2])
    audio = audioSet[dataName]
    if audioAug == True:
        augType = random.randint(0, 1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio,
                                        16000,
                                        numcep=13,
                                        winlen=0.025 * 25 / fps,
                                        winstep=0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage = maxAudio - audio.shape[0]
        audio = numpy.pad(audio, ((0, shortage), (0, 0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)), :]
    return audio


def load_single_audio(audio, fps, numFrames, audioAug=False):
    audio = python_speech_features.mfcc(audio,
                                        16000,
                                        numcep=13,
                                        winlen=0.025 * 25 / fps,
                                        winstep=0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage = maxAudio - audio.shape[0]
        audio = numpy.pad(audio, ((0, shortage), (0, 0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)), :]
    return audio


def load_visual(data, dataPath, numFrames, visualAug):
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg" % faceFolderPath)
    sortedFaceFiles = sorted(faceFiles,
                             key=lambda data: (float(data.split('/')[-1][:-4])),
                             reverse=False)
    faces = []
    H = 112
    if visualAug == True:
        new = int(H * random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H / 2, H / 2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H, H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y + new, x:x + new], (H, H)))
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H, H)))
    faces = numpy.array(faces)
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res


class train_loader(object):

    def __init__(self, cfg, trialFileName, audioPath, visualPath, num_speakers):
        self.cfg = cfg
        self.audioPath = audioPath
        self.visualPath = visualPath
        self.candidate_speakers = num_speakers
        self.path = os.path.join(cfg.DATA.dataPathAVA, "csv")
        self.entity_data = json.load(open(os.path.join(self.path, 'train_entity.json')))
        self.ts_to_entity = json.load(open(os.path.join(self.path, 'train_ts.json')))
        self.mixLst = open(trialFileName).read().splitlines()
        self.list_length = len(self.mixLst)
        random.shuffle(self.mixLst)

    def load_single_audio(self, audio, fps, numFrames, audioAug=False, aug_audio=None):
        if audioAug:
            augType = random.randint(0, 1)
            if augType == 1:
                audio = overlap(audio, aug_audio)
            else:
                audio = audio

        res = vggish_input.waveform_to_examples(audio, 16000, numFrames, fps, return_tensor=False)
        return res

    def load_visual_label_mask(self, videoName, entityName, target_ts, context_ts, visualAug=True):

        faceFolderPath = os.path.join(self.visualPath, videoName, entityName)

        faces = []
        H = 112
        if visualAug == True:
            new = int(H * random.uniform(0.7, 1))
            x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
            M = cv2.getRotationMatrix2D((H / 2, H / 2), random.uniform(-15, 15), 1)
            augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
        else:
            augType = 'orig'
        labels_dict = self.entity_data[videoName][entityName]
        labels = numpy.zeros(len(target_ts))
        mask = numpy.zeros(len(target_ts))

        for i, time in enumerate(target_ts):
            if time not in context_ts:
                faces.append(numpy.zeros((H, H)))
            else:
                labels[i] = labels_dict[time]
                mask[i] = 1
                time = "%.2f" % float(time)
                faceFile = os.path.join(faceFolderPath, str(time) + '.jpg')

                face = cv2.imread(faceFile)

                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (H, H))
                if augType == 'orig':
                    faces.append(face)
                elif augType == 'flip':
                    faces.append(cv2.flip(face, 1))
                elif augType == 'crop':
                    faces.append(cv2.resize(face[y:y + new, x:x + new], (H, H)))
                elif augType == 'rotate':
                    faces.append(cv2.warpAffine(face, M, (H, H)))
        faces = numpy.array(faces)
        return faces, labels, mask

    def get_speaker_context(self, videoName, target_entity, all_ts, center_ts):

        context_speakers = list(self.ts_to_entity[videoName][center_ts])
        context = {}
        chosen_speakers = []
        context[target_entity] = all_ts
        context_speakers.remove(target_entity)
        num_frames = len(all_ts)
        for candidate in context_speakers:
            candidate_ts = self.entity_data[videoName][candidate]
            shared_ts = set(all_ts).intersection(set(candidate_ts))
            if (len(shared_ts) > (num_frames / 2)):
                context[candidate] = shared_ts
                chosen_speakers.append(candidate)
        context_speakers = chosen_speakers
        random.shuffle(context_speakers)
        if not context_speakers:
            context_speakers.insert(0, target_entity)    # make sure is at 0
            while len(context_speakers) < self.candidate_speakers:
                context_speakers.append(random.choice(context_speakers))
        elif len(context_speakers) < self.candidate_speakers:
            context_speakers.insert(0, target_entity)    # make sure is at 0
            while len(context_speakers) < self.candidate_speakers:
                context_speakers.append(random.choice(context_speakers[1:]))
        else:
            context_speakers.insert(0, target_entity)    # make sure is at 0
            context_speakers = context_speakers[:self.candidate_speakers]

        assert set(context_speakers).issubset(set(list(context.keys()))), target_entity
        assert target_entity in context_speakers, target_entity

        return context_speakers, context

    def __getitem__(self, index):

        target_video = self.mixLst[index]
        data = target_video.split('\t')
        fps = float(data[2])
        videoName = data[0][:11]
        target_entity = data[0]
        all_ts = list(self.entity_data[videoName][target_entity].keys())
        numFrames = int(data[1])
        assert numFrames == len(all_ts)

        center_ts = all_ts[math.floor(numFrames / 2)]

        # get context speakers which have more than half time overlapped with target speaker
        context_speakers, context = self.get_speaker_context(videoName, target_entity, all_ts,
                                                             center_ts)

        if self.cfg.TRAIN.AUDIO_AUG:
            other_indices = list(range(0, index)) + list(range(index + 1, self.list_length))
            augment_entity = self.mixLst[random.choice(other_indices)]
            augment_data = augment_entity.split('\t')
            augment_entity = augment_data[0]
            augment_videoname = augment_data[0][:11]
            aug_sr, aug_audio = wavfile.read(
                os.path.join(self.audioPath, augment_videoname, augment_entity + '.wav'))
        else:
            aug_audio = None

        audio_path = os.path.join(self.audioPath, videoName, target_entity + '.wav')
        sr, audio = wavfile.read(os.path.join(self.audioPath, videoName, target_entity + '.wav'))
        audio = self.load_single_audio(audio,
                                       fps,
                                       numFrames,
                                       audioAug=self.cfg.TRAIN.AUDIO_AUG,
                                       aug_audio=aug_audio)

        visualFeatures, labels, masks = [], [], []

        # target_label = list(self.entity_data[videoName][target_entity].values())
        visual, target_labels, target_masks = self.load_visual_label_mask(
            videoName, target_entity, all_ts, all_ts)

        for idx, context_entity in enumerate(context_speakers):
            if context_entity == target_entity:
                label = target_labels
                visualfeat = visual
                mask = target_masks
            else:
                visualfeat, label, mask = self.load_visual_label_mask(videoName, context_entity,
                                                                      all_ts,
                                                                      context[context_entity])
            visualFeatures.append(visualfeat)
            labels.append(label)
            masks.append(mask)

        audio = torch.FloatTensor(audio)[None, :, :]
        visualFeatures = torch.FloatTensor(numpy.array(visualFeatures))
        audio_t = audio.shape[1]
        video_t = visualFeatures.shape[1]
        if audio_t != video_t * 4:
            print(visualFeatures.shape, audio.shape, videoName, target_entity, numFrames)
        labels = torch.LongTensor(numpy.array(labels))
        masks = torch.LongTensor(numpy.array(masks))
        return audio, visualFeatures, labels, masks

    def __len__(self):
        return len(self.mixLst)


class val_loader(object):

    def __init__(self, cfg, trialFileName, audioPath, visualPath, num_speakers):
        self.cfg = cfg
        self.audioPath = audioPath
        self.visualPath = visualPath
        self.candidate_speakers = num_speakers
        self.path = os.path.join(cfg.DATA.dataPathAVA, "csv")
        self.entity_data = json.load(open(os.path.join(self.path, 'val_entity.json')))
        self.ts_to_entity = json.load(open(os.path.join(self.path, 'val_ts.json')))
        self.mixLst = open(trialFileName).read().splitlines()

    def load_single_audio(self, audio, fps, numFrames, audioAug=False, aug_audio=None):

        res = vggish_input.waveform_to_examples(audio, 16000, numFrames, fps, return_tensor=False)
        return res

    def load_visual_label_mask(self, videoName, entityName, target_ts, context_ts):

        faceFolderPath = os.path.join(self.visualPath, videoName, entityName)

        faces = []
        H = 112
        labels_dict = self.entity_data[videoName][entityName]
        labels = numpy.zeros(len(target_ts))
        mask = numpy.zeros(len(target_ts))

        for i, time in enumerate(target_ts):
            if time not in context_ts:
                faces.append(numpy.zeros((H, H)))
            else:
                labels[i] = labels_dict[time]
                mask[i] = 1
                time = "%.2f" % float(time)
                faceFile = os.path.join(faceFolderPath, str(time) + '.jpg')

                face = cv2.imread(faceFile)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (H, H))
                faces.append(face)
        faces = numpy.array(faces)
        return faces, labels, mask

    def get_speaker_context(self, videoName, target_entity, all_ts, center_ts):

        context_speakers = list(self.ts_to_entity[videoName][center_ts])
        context = {}
        chosen_speakers = []
        context[target_entity] = all_ts
        context_speakers.remove(target_entity)
        num_frames = len(all_ts)
        for candidate in context_speakers:
            candidate_ts = self.entity_data[videoName][candidate]
            shared_ts = set(all_ts).intersection(set(candidate_ts))
            context[candidate] = shared_ts
            chosen_speakers.append(candidate)
            # if (len(shared_ts) > (num_frames / 2)):
            # context[candidate] = shared_ts
            # chosen_speakers.append(candidate)
        context_speakers = chosen_speakers
        random.shuffle(context_speakers)
        if not context_speakers:
            context_speakers.insert(0, target_entity)    # make sure is at 0
            while len(context_speakers) < self.candidate_speakers:
                context_speakers.append(random.choice(context_speakers))
        elif len(context_speakers) < self.candidate_speakers:
            context_speakers.insert(0, target_entity)    # make sure is at 0
            while len(context_speakers) < self.candidate_speakers:
                context_speakers.append(random.choice(context_speakers[1:]))
        else:
            context_speakers.insert(0, target_entity)    # make sure is at 0
            context_speakers = context_speakers[:self.candidate_speakers]

        assert set(context_speakers).issubset(set(list(context.keys()))), target_entity

        return context_speakers, context

    def __getitem__(self, index):

        target_video = self.mixLst[index]
        data = target_video.split('\t')
        fps = float(data[2])
        videoName = data[0][:11]
        target_entity = data[0]
        all_ts = list(self.entity_data[videoName][target_entity].keys())
        numFrames = int(data[1])
        # print(numFrames, len(all_ts))
        assert numFrames == len(all_ts)

        center_ts = all_ts[math.floor(numFrames / 2)]

        # get context speakers which have more than half time overlapped with target speaker
        context_speakers, context = self.get_speaker_context(videoName, target_entity, all_ts,
                                                             center_ts)

        sr, audio = wavfile.read(os.path.join(self.audioPath, videoName, target_entity + '.wav'))
        audio = self.load_single_audio(audio, fps, numFrames, audioAug=False)

        visualFeatures, labels, masks = [], [], []

        # target_label = list(self.entity_data[videoName][target_entity].values())
        target_visual, target_labels, target_masks = self.load_visual_label_mask(
            videoName, target_entity, all_ts, all_ts)

        for idx, context_entity in enumerate(context_speakers):
            if context_entity == target_entity:
                label = target_labels
                visualfeat = target_visual
                mask = target_masks
            else:
                visualfeat, label, mask = self.load_visual_label_mask(videoName, context_entity,
                                                                      all_ts,
                                                                      context[context_entity])
            visualFeatures.append(visualfeat)
            labels.append(label)
            masks.append(mask)

        audio = torch.FloatTensor(audio)[None, :, :]
        visualFeatures = torch.FloatTensor(numpy.array(visualFeatures))
        audio_t = audio.shape[1]
        video_t = visualFeatures.shape[1]
        if audio_t != video_t * 4:
            print(visualFeatures.shape, audio.shape, videoName, target_entity, numFrames)
        labels = torch.LongTensor(numpy.array(labels))
        masks = torch.LongTensor(numpy.array(masks))

        return audio, visualFeatures, labels, masks

    def __len__(self):
        return len(self.mixLst)
