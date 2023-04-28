import os
import random
import math


def temporal_batching_index(fr,length=16):
    '''
    Do padding or half-overlapping clips for video.
    
    Input:
        fr: number of frames
    Output:
        batch_indices: array for batch where each element is frame index 
    '''
    if fr < length: 
        #e.g. (1,2,3,4,5) to (1,1,....,1,2,3,4,5,5,...,5,5)
        right = int((length-fr)/2)
        left = length - right - fr
        return [[0]*left + list(range(fr)) + [fr-1]*right]
    
    batch_indices = []
    last_idx = fr - 1
    assert length%2 == 0
    half = int(length/2)
    for i in range(0,fr-half,half):
            frame_indices = [0,]*length
            for j in range(length):
                current_idx =  i + j 
                if current_idx < last_idx:
                    frame_indices[j] = current_idx
                else:
                    frame_indices[j] = last_idx
            batch_indices.append(frame_indices)
            
    return batch_indices

def temporal_sliding_window(clip,window = 16):
    '''
    Make a batched tensor with 16 frame sliding window with the overlap of 8. 
    If a clip is not the multiply of 8, it's padded with the last frames. (1,2...,13,14,14,14) for (1,..,14) 
    If a clip is less than 16 frames, padding is applied like (1,1,....,1,2,3,4,5,5,...,5,5) for (1,2,3,4,5)
    This can be used for sliding window evaluation.
    
    Input:  list of image paths
    Output: torch tensor of shape of (batch,ch,16,h,w).
    '''

    batch_indices = temporal_batching_index(len(clip),length = window)
    
    return [[clip[idx] for idx in  indices] for indices in batch_indices]

def temporal_center_crop(clip,length = 16):
    '''
    Input:  list of image paths
    Output: torch tensor of shape of (1,ch,16,h,w).
    '''
    fr = len(clip) 
    if fr < length: 
        #e.g. (1,2,3,4,5) to (1,1,....,1,2,3,4,5,5,...,5,5)
        right = int((length-fr)/2)
        left = length - right - fr
        indicies =  [0]*left + list(range(fr)) + [fr-1]*right
        output =  [clip[i] for i in indicies]
    elif fr==length:
        output =  clip    
    else:
        middle = int(fr/2)
        assert length%2 == 0
        half = int(length/2)
        start = middle - half
        output =  clip[start : start+length]
        
    return output[::2]



def random_temporal_crop(clip,length = 16):
    '''
    Just randomly sample 16 consecutive frames
    if less than 16 frames, just add padding.
    '''
    fr = len(clip) 
    if fr < length: 
        #e.g. (1,2,3,4,5) to (1,1,....,1,2,3,4,5,5,...,5,5)
        right = int((length-fr)/2)
        left = length - right - fr
        indicies =  [0]*left + list(range(fr)) + [fr-1]*right
        output =  [clip[i] for i in indicies]
    elif fr==length:
        output =  clip
    else:
        start=random.randint(0,fr-length)
        output =  clip[start : start+length]
    return output[::2]


def use_all_frames(clip):
    '''
    Just use it as it is :)
    '''
    return clip

def looppadding(clip, length=16):


        out = clip

        for index in out:
            if len(out) >= length:
                break
            out.append(index)

        return out[::2]

def temporal_even_crop(clip, length=16, n_samples=1):

        clip = list(clip)
        n_frames = len(clip)
        indices = list(range(len(clip)))
        stride = max(
            1, math.ceil((n_frames - 1 - length) / (n_samples - 1)))

        out = []
        for begin_index in indices[::stride]:
            if len(out) >= n_samples:
                break
            end_index = min(indices[-1] + 1, begin_index + length)
            sample = list(range(begin_index, end_index))

            if len(sample) < length:
                out.append([clip[i] for i in looppadding(sample, length=length)])
               # out.append(clip[looppadding(sample, length=length)])
                break
            else:
                out.append([clip[i] for i in sample[::2]])
               # out.append(clip[sample[::2]])

        return out


class TemporalTransform(object):
    def __init__(self,length,mode="center"):
        self.mode = mode
        self.length = length
        #pass dummpy in order to catch incoored mode
        self.__call__(range(128))
        
    def __call__(self, clip):
        if self.mode == "random":
            return random_temporal_crop(clip,self.length)
        elif self.mode == "center":
            return temporal_center_crop(clip,self.length)
        elif self.mode == "all" or self.mode == "nocrop":
            #note that length cannot be satisfied!
            return use_all_frames(clip)
        elif self.mode == "slide":
            #note that output has one more dimention
            return temporal_sliding_window(clip,self.length)
        elif self.mode == "even":
            return temporal_even_crop(clip, self.length, n_samples=5)
        else:
            raise NotImplementedError("this option is not defined:",self.mode)