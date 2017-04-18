from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from models import G,D, weights_init

from tensorboard_logger import configure, log_value


import numpy as np

import datetime


# python train.py --dataset aesthetics-unscaled --cuda --batchSize 1 --testBatchSize 1
# configure("runs/aesthetics-{}".format(datetime.datetime.now()))

# Training settings
parser = argparse.ArgumentParser(description='BEGAN-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='CelebA', default='CelebA')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--h', type=int, default=128, help="h value ( size of noise vector )")
parser.add_argument('--n', type=int, default=128, help="n value")
parser.add_argument('--lambda_k', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.5)
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"



train_set = get_training_set(root_path + opt.dataset)

# test_set = get_test_set(root_path + opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.netG:
    netG = torch.load(opt.netG)
    print('==> Loaded model.')
    for parameter in netG:
        parameter.requires_grad = True
else:
    netG = G(h=opt.h, n=opt.n, output_dim=(3,64,64))
    netG.apply(weights_init)

if opt.netD:
    netD = torch.load(opt.netD)
    print('==> Loaded model.')
    for parameter in netG:
        parameter.requires_grad = True
else:
    netD = D(h=opt.h, n=opt.n, input_dim=(3,64,64))
    netD.apply(weights_init)


print(netG)
print(netD)


criterion_l1 = nn.L1Loss()
real_A = torch.FloatTensor(opt.batchSize, 3, 64, 64)
z_D = torch.FloatTensor(opt.batchSize, opt.h)
z_G = torch.FloatTensor(opt.batchSize, opt.h)

if opt.cuda:
    netG = netG.cuda()
    netD = netD.cuda()
    criterion_l1 = criterion_l1.cuda()
    real_A = real_A.cuda()
    z_D, z_G = z_D.cuda(), z_G.cuda()

real_A = Variable(real_A)
z_D = Variable(z_D)
z_G = Variable(z_G)

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)

total_iterations=0
k_t=0
def train(epoch):
    global total_iterations, k_t
    for iteration, batch in enumerate(training_data_loader, 1):
        real_a_cpu = batch

        ## GT Image
        real_A.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)

        
        netD.zero_grad()
        netG.zero_grad()


        z_D.data.normal_(-1,1)

        G_zD = netG(z_D)

        AE_x = netD(real_A)
        AE_G_zD = netD(G_zD)

        d_loss_real = torch.mean(torch.sum(torch.abs(real_A - AE_x), 1))#criterion_l1(AE_x, real_A)
        d_loss_fake = torch.mean(torch.sum(torch.abs(G_zD - AE_G_zD), 1))#criterion_l1(AE_G_zD, G_zD.detach())

        D_loss = d_loss_real - k_t * d_loss_fake
        D_loss.backward()
        optimizerD.step()




        netD.zero_grad()
        netG.zero_grad()

        z_G.data.normal_(-1,1)

        G_z_G = netG(z_G)
        AE_G_zG = netD(G_z_G)

        G_loss = torch.mean(torch.sum(torch.abs(G_z_G - AE_G_zG), 1))#criterion_l1(G_z_G, AE_G_zG.detach())
        G_loss.backward()
        optimizerG.step()

        g_d_balance = ( opt.gamma * d_loss_real - G_loss ).data[0]

        k_t += opt.lambda_k * g_d_balance
        k_t = max(min(1, k_t), 0)

        global_measure = d_loss_real.data[0] + abs(g_d_balance)

        if True: #iteration % 1000 == 1:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} k_t: {:.4f}".format(
                epoch, iteration, len(training_data_loader), global_measure,k_t))


        if iteration % 100 == 1:
            vutils.save_image(AE_G_zG.data, 'log/{}_AEGzG.jpg'.format(total_iterations), normalize=True)
            total_iterations += iteration

        if iteration % 200 == 0:
            
            # log_value('Loss', loss.data[0], total_iterations)
    # log_value('training_loss', loss.data[0], epoch)


def test(epoch):
    pass




for epoch in range(1, opt.nEpochs + 1):
    #net.train()
    train(epoch)
    # net.eval()
    # test(epoch)


    if epoch % 5 == 0:
        checkpoint(epoch)
