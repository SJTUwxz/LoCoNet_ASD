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
python train.py --dataPathAVA AVADataPath --download 
```

`AVADataPath` is the folder you want to save the AVA dataset and its preprocessing outputs, the details can be found in [here](https://github.com/TaoRuijie/TalkNet_ASD/blob/main/utils/tools.py#L34) . Please read them carefully.

After AVA dataset is downloaded, please change the DATA.dataPathAVA entry in the config file. 

#### Training script
```
python -W ignore::UserWarning train.py --cfg configs/multi.yaml OUTPUT_DIR <output directory>
```



#### Pretrained model

Please download the LoCoNet trained weights on AVA dataset [here](https://drive.google.com/file/d/1EX-V464jCD6S-wg68yGuAa-UcsMrw8mK/view?usp=sharing).

```
python -W ignore::UserWarning test_multicard.py --cfg configs/multi.yaml  RESUME_PATH {model download path}
```

### Citation

Please cite the following if our paper or code is helpful to your research.
```
@article{wang2023loconet,
  title={LoCoNet: Long-Short Context Network for Active Speaker Detection},
  author={Wang, Xizi and Cheng, Feng and Bertasius, Gedas and Crandall, David},
  journal={arXiv preprint arXiv:2301.08237},
  year={2023}
}
```


### Acknowledge

The code base of this project is studied from [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) which is a very easy-to-use ASD pipeline.


