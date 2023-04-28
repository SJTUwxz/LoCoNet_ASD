# How to process video as data loader

We assume that video is preprocessed in to image files in advance. Usually, we do not use all frames in a clip but sample a certain duration (e.g. 16 frames). The pipline we assume for each chunk is the following.

- Get a list of images paths of clips e.g. ["./video/clip1/frame0.jpg",...,"./video/clip1/frame101.jpg"]
- Sample a certain duration we want to use  e.g. ["./video/clip1/frame11.jpg",...,"./video/clip1/frame26.jpg"]
- Load each frames into a tensor shaped as (T, H, W, C). HW can be changed later. 
- Use torchvision builtin utilities to crop, flip, etc. For example, 
    - ToTensorVideo() from (T, H, W, C) to (C, T, H, W)), from 0-255 to 0-1 (devide by 225), and from uint8 to float.   
    - CenterCropVideo
    - RandomHorizontalFlipVideo
    - NormalizeVideo with kinetics mean and std
    -See more https://github.com/pytorch/vision/blob/f0d3daa7f65bcde560e242d9bccc284721368f02/torchvision/transforms/transforms_video.py

Note that the first part is different from what official pytorch repository ( https://github.com/pytorch/vision/tree/master/references/video_classification ) does. We don't use VideoClip class.