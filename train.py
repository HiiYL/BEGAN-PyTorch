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
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate. Default=0.001')
parser.add_argument('--lr_update_step', type=float, default=10000, help='Reduce learning rate by factor of 2 every n iterations. Default=1')
parser.add_argument('--cuda', action='store_true', help='use cuda?')

parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--log_step', default=10, help="logging frequency")
parser.add_argument('--image_size', default=128, help="image size")


parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--h', type=int, default=512, help="h value ( size of noise vector )")
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
    #transforms.CenterCrop(160),
    transforms.Scale(size=(opt.image_size,opt.image_size)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



train_set = DatasetFromFolder(join(join(root_path,opt.dataset), "train"), train_transform)
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
    netG = G(h=opt.h, n=opt.n, output_dim=(3,opt.image_size,opt.image_size))
    netG.apply(weights_init)

if opt.netD:
    netD = torch.load(opt.netD)
    print('==> Loaded model.')
    for parameter in netG:
        parameter.requires_grad = True
else:
    netD = D(h=opt.h, n=opt.n, input_dim=(3,opt.image_size,opt.image_size))
    netD.apply(weights_init)

print(netG)
print(netD)
criterion_l1 = nn.L1Loss()
real_A = torch.FloatTensor(opt.batchSize, 3, opt.image_size, opt.image_size)
embedding_v = torch.FloatTensor(opt.batchSize, 1024)
wrong_embedding_v = torch.FloatTensor(opt.batchSize, 1024)
z_D = torch.FloatTensor(opt.batchSize, opt.h)
z_G = torch.FloatTensor(opt.batchSize, opt.h)

if opt.cuda:
    netG = netG.cuda()
    netD = netD.cuda()
    criterion_l1 = criterion_l1.cuda()
    embedding_v = embedding_v.cuda()
    wrong_embedding_v = wrong_embedding_v.cuda()
    real_A = real_A.cuda()
    z_D, z_G = z_D.cuda(), z_G.cuda()


embedding_v = Variable(embedding_v)
real_A = Variable(real_A)
z_D = Variable(z_D)
z_G = Variable(z_G)

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

total_iterations=0
k_t=0
fixed_sample = None
fixed_x = None
fixed_embedding = None
def train(epoch, save_path, total_iterations, k_t, fixed_sample, fixed_x, fixed_embedding):
    for iteration, batch in enumerate(training_data_loader, 1):
        real_a_cpu, embedding, wrong_embedding = batch
        ## GT Image
        real_A.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        embedding_v.data.resize_(embedding.size()).copy_(embedding)

        wrong_embedding_v.data.resize_(wrong_embedding.size()).copy_(wrong_embedding)

        
        netD.zero_grad()
        netG.zero_grad()


        current_batch_size = embedding.size(0)


        z_D.data.resize_(current_batch_size, z_D.size(1)).normal_(0,1)
        z_G.data.resize_(current_batch_size, z_G.size(1)).normal_(0,1)

        G_zD = netG(torch.cat((embedding_v,z_D),1))
        AE_x = netD(real_A,embedding_v)
        AE_G_zD = netD(G_zD.detach())

        G_zG = netG(torch.cat((embedding_v,z_G),1))
        AE_G_zG = netD(G_zG,embedding_v)

        AE_x_wrong = netD(real_A, wrong_embedding_v)

        d_loss_real = torch.mean(torch.abs(AE_x - real_A))#criterion_l1(AE_x, real_A) #
        d_loss_wrong_comment = torch.mean(torch.abs(AE_x_wrong - real_A))
        d_loss_fake = torch.mean(torch.abs(AE_G_zD - G_zD)) #criterion_l1(AE_G_zD, G_zD.detach()) ##

        D_loss = d_loss_real - d_loss_wrong_comment - k_t * d_loss_fake
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
            fixed_embedding = Variable(embedding_v.clone().data, volatile=True)
            vutils.save_image(real_A.data, '{}/x_fixed.jpg'.format(save_path), normalize=True,range=(-1,1))

        balance = ( opt.gamma * d_loss_real - G_loss ).data[0]
        measure = d_loss_real.data[0] + abs(balance)

        k_t += opt.lambda_k * balance
        k_t = max(min(1, k_t), 0)

        total_iterations += 1

        if total_iterations % 10 == 0:
            print("===> Epoch[{}]({}/{}): D_Loss: {:.4f} | G_Loss: {:.4f} | Measure: {:.4f} | k_t: {:.4f}".format(
                epoch, iteration, len(training_data_loader), D_loss.data[0], G_loss.data[0], measure,k_t))

        if total_iterations % 100 == 0:
            log_value('D_Loss', D_loss.data[0], total_iterations)
            log_value('G_Loss', G_loss.data[0], total_iterations)
            log_value('Measure', measure, total_iterations)
            log_value('k', k_t, total_iterations)

        if (total_iterations % 500 == 0) or total_iterations == 1:
            ae_x = netD(fixed_x,fixed_embedding)
            g = netG(torch.cat((fixed_embedding,fixed_sample),1))
            ae_g = netD(g,fixed_embedding)
            
            vutils.save_image(ae_g.data, '{}/{}_D_fake.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))
            vutils.save_image(ae_x.data, '{}/{}_D_real.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))
            vutils.save_image(g.data, '{}/{}_G.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))

        if total_iterations % opt.lr_update_step == 0:
            lr =  opt.lr * (0.5 ** (total_iterations//opt.lr_update_step))
            print("reducing lr to {} at iteration {}".format(lr, total_iterations))
            for param_group in optimizerG.param_groups:
                param_group['lr'] = lr
            for param_group in optimizerD.param_groups:
                param_group['lr'] = lr

    return total_iterations, k_t, fixed_sample, fixed_x, fixed_embedding

def test(epoch):
    pass

def checkpoint(epoch, save_path):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))

    now = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
    netG_model_out_path = "{}/netG_model_epoch_{}.pth".format(save_path,epoch)
    netD_model_out_path = "{}/netD_model_epoch_{}.pth".format(save_path,epoch)
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
for epoch in range(1, opt.nEpochs + 1):
    netG.train()
    netD.train()
    total_iterations, k_t, fixed_sample, fixed_x, fixed_embedding = train(epoch, save_path, total_iterations, k_t, fixed_sample, fixed_x, fixed_embedding)
    # net.eval()
    # test(epoch)

    if epoch % 10 == 0:
        checkpoint(epoch, save_path)
