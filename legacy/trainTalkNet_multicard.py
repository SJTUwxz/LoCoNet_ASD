import time, os, torch, argparse, warnings, glob

from utils.tools import *
from dlhammer import bootstrap
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class MyCollator(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, data):
        audiofeatures = [item[0] for item in data]
        visualfeatures = [item[1] for item in data]
        labels = [item[2] for item in data]
        masks = [item[3] for item in data]
        cut_limit = self.cfg.MODEL.CLIP_LENGTH
        # pad audio
        lengths = torch.tensor([t.shape[1] for t in audiofeatures])
        max_len = max(lengths)
        padded_audio = torch.stack([
            torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1], i.shape[2]))], 1)
            for i in audiofeatures
        ], 0)

        if max_len > cut_limit * 4:
            padded_audio = padded_audio[:, :, :cut_limit * 4, ...]

        # pad video
        lengths = torch.tensor([t.shape[1] for t in visualfeatures])
        max_len = max(lengths)
        padded_video = torch.stack([
            torch.cat(
                [i, i.new_zeros((i.shape[0], max_len - i.shape[1], i.shape[2], i.shape[3]))], 1)
            for i in visualfeatures
        ], 0)
        padded_labels = torch.stack(
            [torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1]))], 1) for i in labels], 0)
        padded_masks = torch.stack(
            [torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1]))], 1) for i in masks], 0)

        if max_len > cut_limit:
            padded_video = padded_video[:, :, :cut_limit, ...]
            padded_labels = padded_labels[:, :, :cut_limit, ...]
            padded_masks = padded_masks[:, :, :cut_limit, ...]
        return padded_audio, padded_video, padded_labels, padded_masks


class DataPrep(pl.LightningDataModule):

    def __init__(self, cfg):
        self.cfg = cfg

    def train_dataloader(self):
        cfg = self.cfg

        if self.cfg.MODEL.NAME == "baseline":
            from dataLoader import train_loader, val_loader
            loader = train_loader(trialFileName = cfg.trainTrialAVA, \
                              audioPath      = os.path.join(cfg.audioPathAVA , 'train'), \
                              visualPath     = os.path.join(cfg.visualPathAVA, 'train'), \
                              batchSize=2500
                              )
        elif self.cfg.MODEL.NAME == "multi":
            from dataLoader_multiperson import train_loader, val_loader
            loader = train_loader(trialFileName = cfg.trainTrialAVA, \
                              audioPath      = os.path.join(cfg.audioPathAVA , 'train'), \
                              visualPath     = os.path.join(cfg.visualPathAVA, 'train'), \
                              num_speakers=cfg.MODEL.NUM_SPEAKERS,
                              )
        if cfg.MODEL.NAME == "baseline":
            trainLoader = torch.utils.data.DataLoader(
                loader,
                batch_size=1,
                shuffle=True,
                num_workers=4,
            )
        elif cfg.MODEL.NAME == "multi":
            collator = MyCollator(cfg)
            trainLoader = torch.utils.data.DataLoader(loader,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      collate_fn=collator)

        return trainLoader

    def val_dataloader(self):
        cfg = self.cfg
        loader = val_loader(trialFileName = cfg.evalTrialAVA, \
                            audioPath     = os.path.join(cfg.audioPathAVA , cfg.evalDataType), \
                            visualPath    = os.path.join(cfg.visualPathAVA, cfg.evalDataType), \
                            )
        valLoader = torch.utils.data.DataLoader(loader,
                                                batch_size=cfg.VAL.BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=16)
        return valLoader


def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    cfg = bootstrap(print_cfg=False)
    print(cfg)

    warnings.filterwarnings("ignore")
    seed_everything(42, workers=True)

    cfg = init_args(cfg)

    # checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(cfg.WORKSPACE, "model"),
    #                                       save_top_k=-1,
    #                                       filename='{epoch}')

    data = DataPrep(cfg)

    trainer = Trainer(
        gpus=int(cfg.TRAIN.TRAINER_GPU),
        precision=32,
    # callbacks=[checkpoint_callback],
        max_epochs=25,
        replace_sampler_ddp=True)
    # val_trainer = Trainer(deterministic=True, num_sanity_val_steps=-1, gpus=1)
    if cfg.downloadAVA == True:
        preprocess_AVA(cfg)
        quit()

    # if cfg.RESUME:
    #     modelfiles = glob.glob('%s/model_0*.model' % cfg.modelSavePath)
    #     modelfiles.sort()
    #     if len(modelfiles) >= 1:
    #         print("Model %s loaded from previous state!" % modelfiles[-1])
    #         epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
    #         s = talkNet(cfg)
    #         s.loadParameters(modelfiles[-1])
    #     else:
    #         epoch = 1
    #         s = talkNet(cfg)
    epoch = 1
    if cfg.MODEL.NAME == "baseline":
        from talkNet_multicard import talkNet
    elif cfg.MODEL.NAME == "multi":
        from talkNet_multi import talkNet

    s = talkNet(cfg)

    # scoreFile = open(cfg.scoreSavePath, "a+")

    trainer.fit(s, train_dataloaders=data.train_dataloader())

    modelfiles = glob.glob('%s/*.pth' % os.path.join(cfg.WORKSPACE, "model"))

    modelfiles.sort()
    for path in modelfiles:
        s.loadParameters(path)
        prec = trainer.validate(s, data.val_dataloader())

    # if epoch % cfg.testInterval == 0:
    # s.saveParameters(cfg.modelSavePath + "/model_%04d.model" % epoch)
    # trainer.validate(dataloaders=valLoader)
    # print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%" % (epoch, mAPs[-1]))
    # scoreFile.write("%d epoch, LOSS %f, mAP %2.2f%%\n" % (epoch, loss, mAPs[-1]))
    # scoreFile.flush()


if __name__ == '__main__':
    main()
