from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import DatasetFromFolder
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from models import G,D, weights_init

from torchvision import transforms
from os.path import join
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
parser.add_argument('--lr_update_step', type=float, default=100000, help='Reduce learning rate by factor of 2 every n iterations. Default=1')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--poll_step', default=1000, help="how often to poll if training has plateaued")
parser.add_argument('--patience', default=10, help="how long to wait before reducing lr")


parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--log_step', default=10, help="logging frequency")
parser.add_argument('--tb_log_step', default=100, help="tensorboard logging frequency")
parser.add_argument('--visualize_step', default=500, help="display image frequency")
parser.add_argument('--checkpoint_step', default=50000, help="checkpoint frequency")


parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--h', type=int, default=64, help="h value ( size of noise vector )")
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

train_transform = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Scale(size=64),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



train_set = DatasetFromFolder(join(join(root_path,opt.dataset), "train"), train_transform)
# test_set = get_test_set(root_path + opt.dataset)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)


print('===> Building model')
if opt.netG:
    netG = torch.load(opt.netG)
    print('==> Loaded model.')
    for parameter in netG.parameters():
        parameter.requires_grad = True
else:
    netG = G(h=opt.h, n=opt.n, output_dim=(3,64,64))
    netG.apply(weights_init)

if opt.netD:
    netD = torch.load(opt.netD)
    print('==> Loaded model.')
    for parameter in netD.parameters():
        parameter.requires_grad = True
else:
    netD = D(h=opt.h, n=opt.n, input_dim=(3,64,64))
    netD.apply(weights_init)

print(netG)
print(netD)
real_A = torch.FloatTensor(opt.batchSize, 3, 64, 64)
z_D = torch.FloatTensor(opt.batchSize, opt.h)
z_G = torch.FloatTensor(opt.batchSize, opt.h)

if opt.cuda:
    netG = netG.cuda()
    netD = netD.cuda()
    real_A = real_A.cuda()
    z_D, z_G = z_D.cuda(), z_G.cuda()

real_A = Variable(real_A)
z_D = Variable(z_D)
z_G = Variable(z_G)

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


def train(options_to_track):
    total_iterations, k_t, fixed_sample, fixed_x, best_measure,patience = options_to_track
    for iteration, batch in enumerate(training_data_loader, 1):
        real_a_cpu = batch
        ## GT Image
        real_A.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)

        
        netD.zero_grad()
        netG.zero_grad()

        z_D.data.normal_(0,1)
        z_G.data.normal_(0,1)

        G_zD = netG(z_D)
        AE_x = netD(real_A)
        AE_G_zD = netD(G_zD.detach())

        G_zG = netG(z_G)
        AE_G_zG = netD(G_zG)

        d_loss_real = torch.mean(torch.abs(AE_x - real_A))
        d_loss_fake = torch.mean(torch.abs(AE_G_zD - G_zD))

        D_loss = d_loss_real - k_t * d_loss_fake
        D_loss.backward()
        optimizerD.step()

        netD.zero_grad()
        netG.zero_grad()

        G_loss = torch.mean(torch.abs(G_zG - AE_G_zG)) #criterion_l1(G_zG, AE_G_zG.detach())#
        G_loss.backward()

        optimizerG.step()

        if fixed_sample is None:
            fixed_sample = Variable(z_G.clone().data, volatile=True)
            fixed_x = Variable(real_A.clone().data, volatile=True)
            vutils.save_image(real_A.data, '{}/x_fixed.jpg'.format(save_path), normalize=True,range=(-1,1))

        balance = ( opt.gamma * d_loss_real - G_loss ).data[0]
        measure = d_loss_real.data[0] + abs(balance)

        k_t += opt.lambda_k * balance
        k_t = max(min(1, k_t), 0)

        total_iterations += 1

        if total_iterations % opt.log_step == 0:
            print("===> Epoch[{}]({}/{}): D_Loss: {:.4f} | G_Loss: {:.4f} | Measure: {:.4f} | k_t: {:.4f}".format(
                epoch, iteration, len(training_data_loader), D_loss.data[0], G_loss.data[0], measure,k_t))

        if total_iterations % opt.tb_log_step == 0:
            log_value('D_Loss', D_loss.data[0], total_iterations)
            log_value('G_Loss', G_loss.data[0], total_iterations)
            log_value('Measure', measure, total_iterations)
            log_value('k', k_t, total_iterations)

        if (total_iterations % opt.visualize_step == 0) or total_iterations == 1:
            ae_x = netD(fixed_x)
            g = netG(fixed_sample)
            ae_g = netD(g)
            
            vutils.save_image(ae_g.data, '{}/{}_D_fake.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))
            vutils.save_image(ae_x.data, '{}/{}_D_real.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))
            vutils.save_image(g.data, '{}/{}_G.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))

        if total_iterations % opt.checkpoint_step == 0:
            checkpoint(total_iterations, save_path)

        if total_iterations % opt.poll_step == 0:
            if measure < best_measure:
                best_measure = measure
            else:
                patience -= 1
                print("[!] Measure not decreasing | Best : {} | Current: {} | Patience: {}")
                if patience <= 0:
                    patience = opt.patience
                    times_reduced_lr += 1
                    lr =  opt.lr * (0.5 ** times_reduced_lr)
                    print("[!] Reducing lr to {} at iteration {}".format(lr, total_iterations))
                    for param_group in optimizerG.param_groups:
                        param_group['lr'] = lr
                    for param_group in optimizerD.param_groups:
                        param_group['lr'] = lr

    return (total_iterations, k_t, fixed_sample, fixed_x, best_measure,patience)

def test(epoch):
    pass

def checkpoint(epoch, save_path):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))

    now = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
    netG_model_out_path = "{}/netG_model_iter_{}.pth".format(save_path,epoch)
    netD_model_out_path = "{}/netD_model_iter_{}.pth".format(save_path,epoch)
    torch.save(netG, netG_model_out_path)
    torch.save(netD, netD_model_out_path)
    print("Checkpoint saved to {}".format(save_path))


if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists(os.path.join("logs", opt.dataset)):
    os.mkdir(os.path.join("logs", opt.dataset))

now = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
save_path = os.path.join(os.path.join("logs", opt.dataset), now)

if not os.path.exists(save_path):
    os.mkdir(save_path)

configure(save_path)

total_iterations=0
k_t=0
fixed_sample = None
fixed_x = None
best_measure = 1e7
patience = opt.patience
options_to_track = (total_iterations, k_t, fixed_sample, fixed_x, best_measure,patience)
for epoch in range(1, opt.nEpochs + 1):
    netG.train()
    netD.train()
    options_to_track = train(options_to_track)
    # net.eval()
    # test(epoch)
