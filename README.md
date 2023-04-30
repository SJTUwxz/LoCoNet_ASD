## LoCoNet: Long-Short Context Network for Active Speaker Detection



### Dependencies

Start from building the environment
```
conda env create -f requirements.yml
conda activate loconet
```
export PYTHONPATH=**project_dir**/dlhammer:$PYTHONPATH
and replace **project_dir** with your code base location



### Data preparation

We follow TalkNet's data preparation script to download and prepare the AVA dataset.

```
python trainTalkNet.py --dataPathAVA AVADataPath --download 
```

`AVADataPath` is the folder you want to save the AVA dataset and its preprocessing outputs, the details can be found in [here](https://github.com/TaoRuijie/TalkNet_ASD/blob/main/utils/tools.py#L34) . Please read them carefully.

After AVA dataset is downloaded, please change the DATA.dataPathAVA entry in the config file. 

#### Training script
```
python -W ignore::UserWarning trainTalkNet_config.py --cfg configs/multi.yaml OUTPUT_DIR <output directory>
```



#### Pretrained model
Our pretrained model performs `mAP: 95.1` in validation set, you can check it by using: 
```
python trainTalkNet.py --dataPathAVA AVADataPath --evaluation
```


### Citation

Please cite the following if our paper or code is helpful to your research.
```

```
I have summaried some potential [FAQs](https://github.com/TaoRuijie/TalkNet_ASD/blob/main/FAQ.md). You can also check the `issues` in Github for other questions that I have answered.

This is my first open-source work, please let me know if I can future improve in this repositories or there is anything wrong in our work. Thanks for your support!

### Acknowledge




