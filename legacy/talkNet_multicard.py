import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

import pytorch_lightning as pl
from torch import distributed as dist


class talkNet(pl.LightningModule):

    def __init__(self, cfg):
        super(talkNet, self).__init__()
        self.cfg = cfg
        self.model = talkNetModel()
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()
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
        audioFeature, visualFeature, labels = batch
        audioEmbed = self.model.forward_audio_frontend(audioFeature[0])    # feedForward
        visualEmbed = self.model.forward_visual_frontend(visualFeature[0])
        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        outsA = self.model.forward_audio_backend(audioEmbed)
        outsV = self.model.forward_visual_backend(visualEmbed)
        labels = labels[0].reshape((-1))
        nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
        nlossA = self.lossA.forward(outsA, labels)
        nlossV = self.lossV.forward(outsV, labels)
        loss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.saveParameters(
            os.path.join(self.cfg.WORKSPACE, "model", "{}.pth".format(self.current_epoch)))

    def validation_step(self, batch, batch_idx):
        audioFeature, visualFeature, labels, indices = batch
        audioEmbed = self.model.forward_audio_frontend(audioFeature[0])
        visualEmbed = self.model.forward_visual_frontend(visualFeature[0])
        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        labels = labels[0].reshape((-1))
        loss, predScore, _, _ = self.lossAV.forward(outsAV, labels)
        predScore = predScore[:, -1:].detach().cpu().numpy()
        # self.log("val_loss", loss)

        return predScore

    def validation_epoch_end(self, validation_step_outputs):
        evalCsvSave = self.cfg.evalCsvSave
        evalOrig = self.cfg.evalOrig
        predScores = []

        for out in validation_step_outputs:    # batch size =1
            predScores.extend(out)

        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        print(len(evalRes), len(predScores), len(evalLines))
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s " % (evalOrig,
                                                                                      evalCsvSave)
        mAP = float(
            str(subprocess.run(cmd, shell=True, capture_output=True).stdout).split(' ')[2][:5])
        print("validation mAP: {}".format(mAP))

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location='cpu')
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

    def evaluate_network(self, loader):
        self.eval()
        self.model = self.model.cuda()
        self.lossAV = self.lossAV.cuda()
        self.lossA = self.lossA.cuda()
        self.lossV = self.lossV.cuda()
        predScores = []
        evalCsvSave = self.cfg.evalCsvSave
        evalOrig = self.cfg.evalOrig
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(
                    audioEmbed, visualEmbed)
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                labels = labels[0].reshape((-1)).cuda()
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)
                predScore = predScore[:, 1].detach().cpu().numpy()
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
