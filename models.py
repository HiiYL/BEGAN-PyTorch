import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class G(nn.Module):
    def __init__(self, h, n, output_dim=(64,64,3)):
        super(G, self).__init__()
        self.n = n
        self.h = h

        channel, width, height = output_dim
        self.blocks = int(np.log2(width) - 2)

        print("[!] {} blocks in G ".format(self.blocks))

        self.fc = nn.Linear(h + 1024, 8 * 8 * n)

        conv_layers = []
        for i in range(self.blocks):
            conv_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ELU())
            conv_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ELU())

            if i < self.blocks - 1:
                conv_layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        conv_layers.append( nn.Conv2d(n,channel, kernel_size=3, stride=1, padding=1) )
        self.conv = nn.Sequential(*conv_layers)

        #self.tanh = nn.Tanh()


    def forward(self, x):
        fc_out = self.fc(x).view(-1,self.n,8,8)
        return self.conv(fc_out)

class D(nn.Module):
    def __init__(self, h, n, input_dim=(64,64,3)):
        super(D, self).__init__()

        self.n = n
        self.h = h

        channel, width, height = input_dim
        self.blocks = int(np.log2(width) - 2)

        print("[!] {} blocks in D ".format(self.blocks))

        encoder_layers = []
        encoder_layers.append(nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1))

        prev_channel_size = n
        for i in range(self.blocks):
            channel_size = ( i + 1 ) * n
            encoder_layers.append(nn.Conv2d(prev_channel_size, channel_size, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ELU())
            encoder_layers.append(nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ELU())

            if i < self.blocks - 1:
                # Downsampling
                encoder_layers.append(nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=2, padding=1))
                encoder_layers.append(nn.ELU())

            prev_channel_size = channel_size

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_encode = nn.Linear(8 * 8 * self.blocks * n, h + 1024)
        self.fc_decode = nn.Linear(h + 1024, 8 * 8 * n)

        decoder_layers = []
        for i in range(self.blocks):
            decoder_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.ELU())
            decoder_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.ELU())

            if i < self.blocks - 1:
                decoder_layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        decoder_layers.append( nn.Conv2d(n,channel, kernel_size=3, stride=1, padding=1) )
        self.decoder = nn.Sequential(*decoder_layers)
        #self.tanh = nn.Tanh()


    def forward(self,x, embedding):
        #   encoder        [ flatten ] 
        x = self.encoder(x).view(x.size(0), -1)
        # print(x)
        x = self.fc_encode(x)

        x = torch.cat((x, embedding), 1)
        #   decoder
        x = self.fc_decode(x).view(-1,self.n,8,8)
        x = self.decoder(x)

        return x






