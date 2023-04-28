import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss_multi import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

import pytorch_lightning as pl
from torch import distributed as dist


class talkNet(pl.LightningModule):

    def __init__(self, cfg):
        super(talkNet, self).__init__()
        self.model = talkNetModel().cuda()
        self.cfg = cfg
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        print(
            time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
            (sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.SOLVER.BASE_LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=1,
                                                    gamma=self.cfg.SOLVER.SCHEDULER.GAMMA)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        audioFeature, visualFeature, labels, masks = batch
        b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
        audioFeature = audioFeature.repeat(1, s, 1, 1)
        audioFeature = audioFeature.view(b * s, *audioFeature.shape[2:])
        visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
        labels = labels.view(b * s, *labels.shape[2:])
        masks = masks.view(b * s, *masks.shape[2:])

        audioEmbed = self.model.forward_audio_frontend(audioFeature)    # feedForward
        visualEmbed = self.model.forward_visual_frontend(visualFeature)
        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        outsA = self.model.forward_audio_backend(audioEmbed)
        outsV = self.model.forward_visual_backend(visualEmbed)
        labels = labels.reshape((-1))
        nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, masks)
        nlossA = self.lossA.forward(outsA, labels, masks)
        nlossV = self.lossV.forward(outsV, labels, masks)
        loss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.saveParameters(
            os.path.join(self.cfg.WORKSPACE, "model", "{}.pth".format(self.current_epoch)))

    def evaluate_network(self, loader):
        self.eval()
        predScores = []
        self.model = self.model.cuda()
        self.lossAV = self.lossAV.cuda()
        self.lossA = self.lossA.cuda()
        self.lossV = self.lossV.cuda()
        evalCsvSave = self.cfg.evalCsvSave
        evalOrig = self.cfg.evalOrig
        for audioFeature, visualFeature, labels, masks in tqdm.tqdm(loader):
            with torch.no_grad():
                b, s = visualFeature.shape[0], visualFeature.shape[1]
                t = visualFeature.shape[2]
                audioFeature = audioFeature.repeat(1, s, 1, 1)
                audioFeature = audioFeature.view(b * s, *audioFeature.shape[2:])
                visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
                labels = labels.view(b * s, *labels.shape[2:])
                masks = masks.view(b * s, *masks.shape[2:])
                audioEmbed = self.model.forward_audio_frontend(audioFeature.cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature.cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(
                    audioEmbed, visualEmbed)
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                labels = labels.reshape((-1)).cuda()
                outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
                labels = labels.view(b, s, t)[:, 0, :].view(b * t)
                masks = masks.view(b, s, t)[:, 0, :].view(b * t)
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels, masks)
                predScore = predScore.detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s " % (evalOrig,
                                                                                      evalCsvSave)
        mAP = float(
            str(subprocess.run(cmd, shell=True, capture_output=True).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model." % origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s" %
                                 (origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
