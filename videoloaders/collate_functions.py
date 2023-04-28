import torch
from torch.utils.data.dataloader import default_collate
def collate_video(batch):
    '''
    Our video is (temporal_crops, C, T, H, W) where temporal_crops differes from clip to clip
    We can't use standard collate function. 
    Instead of stacking, let's do cat
    Keep in mind that this will also need list of frame length in order to restore each videos later. 
    '''
    elem = batch[0]
    assert isinstance(elem,dict)
    output = {key: default_collate([d[key] for d in batch]) for key in elem if key!='input'}
    output["input"] = torch.cat([d["input"] for d in batch])
    return output
    