import torch
import torch.nn as nn

# from model.visualEncoder import visualFrontend, visualTCN, visualConv1D
from model.attentionLayer import attentionLayer
from model.convLayer import ConvLayer
from torchvggish import vggish
from model.visualEncoder import visualFrontend, visualConv1D, visualTCN


class locoencoder(nn.Module):

    def __init__(self, cfg):
        super(locoencoder, self).__init__()
        self.cfg = cfg
        # Visual Temporal Encoder
        self.visualFrontend = visualFrontend(cfg)    # Visual Frontend
        self.visualTCN = visualTCN()    # Visual Temporal Network TCN
        self.visualConv1D = visualConv1D()    # Visual Temporal Network Conv1d

        urls = {
            'vggish':
                "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
        }
        self.audioEncoder = vggish.VGGish(urls, preprocess=False, postprocess=False)
        self.audio_pool = nn.AdaptiveAvgPool1d(1)

        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model=128, nhead=8)
        self.crossV2A = attentionLayer(d_model=128, nhead=8)

        # Audio-visual Self Attention

        num_layers = self.cfg.MODEL.AV_layers
        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(ConvLayer(cfg))
            layers.append(attentionLayer(d_model=256, nhead=8))
        self.convAV = layers

    def forward_visual_frontend(self, x):

        B, T, W, H = x.shape
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1, 2)
        return x

    def forward_audio_frontend(self, x):
        t = x.shape[-2]
        numFrames = t // 4
        pad = 8 - (t % 8)
        x = torch.nn.functional.pad(x, (0, 0, 0, pad), "constant")
        # x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)

        b, c, t2, freq = x.shape
        x = x.view(b * c, t2, freq)
        x = self.audio_pool(x)
        x = x.view(b, c, t2)[:, :, :numFrames]
        x = x.permute(0, 2, 1)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src=x1, tar=x2, adjust=self.cfg.MODEL.ADJUST_ATTENTION)
        x2_c = self.crossV2A(src=x2, tar=x1, adjust=self.cfg.MODEL.ADJUST_ATTENTION)
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2, b=1, s=1):
        x = torch.cat((x1, x2), 2)    # B*S, T, 2C
        for i, layer in enumerate(self.convAV):
            if i % 2 == 0:
                x, b, s = layer(x, b, s)
            else:
                x = layer(src=x, tar=x)

        x = torch.reshape(x, (-1, 256))
        return x

    def forward_audio_backend(self, x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self, x):
        x = torch.reshape(x, (-1, 128))
        return x
