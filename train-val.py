from __future__ import print_function
import click
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from itertools import chain

from utils.logger import Logger
from utils.utils import is_image_file
from utils.dataset import DATASET
from utils.ImagePool import ImagePool
from model.Discriminator import Discriminator
from model.Generator import Generator

#reference: https://github.com/sunshineatnoon/Paper-Implementations/blob/master/cycleGAN/CycleGAN.py

@click.command()
@click.option('--batchsize', type=int, default=1, help='with batchsize=1 equivalent to instance normalization.')
@click.option('--ngf', type=int, default=64)
@click.option('--ndf', type=int, default=64)
@click.option('--niter', type=int, default=50000, help='number of iterations to train for')
@click.option('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
@click.option('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
@click.option('--weight_decay', type=float, default=1e-4, help='weight decay in network D, default=1e-4')
@click.option('--cuda', default=True, help='enables cuda')
@click.option('--ckp_dir', default='checkpoints/', help='folder to output images and model checkpoints')
@click.option('--loadsize', type=int, default=143, help='scale image to this size')
@click.option('--finesize', type=int, default=128, help='random crop image to this size')
@click.option('--input_nc', type=int, default=3, help='channel number of input image')
@click.option('--output_nc', type=int, default=3, help='channel number of output image')
@click.option('--save_step', type=int, default=10000, help='save interval')
@click.option('--test_step', type=int, default=250, help='test interval')
@click.option('--log_step', type=int, default=100, help='log interval')
@click.option('--loss_type', default='mse', help='GAN loss type, bce|mse default is negative likelihood loss')
@click.option('--poolsize', type=int, default=50, help='size of buffer in lsGAN, poolsize=0 indicates not using history')
@click.option('--path_g_ab', default='', help='path to pre-trained G_AB')
@click.option('--path_g_ba', default='', help='path to pre-trained G_BA')

def main(batchsize, ngf, ndf, niter, lr, beta1, weight_decay, cuda, ckp_dir, loadsize, finesize,
        input_nc, output_nc, save_step, test_step, log_step, loss_type, poolsize, path_g_ab, path_g_ba):
    try:
        os.makedirs(ckp_dir)
        os.makedirs('imgs/')
    except OSError:
        pass

    Loggers = {'Loss_D':Logger('./logs/Loss_D'),
               'Loss_G':Logger('./logs/Loss_G'),
               'Loss_MSE':Logger('./logs/Loss_MSE'),
               'img_log':Logger('./logs/img_log')}

    torch.cuda.manual_seed_all(random.randint(1, 10000))
    cudnn.benchmark = True

    ##########   DATASET   ###########
    datasetA = DATASET(r'D:\workspace\dataset\pixiv_coloring\imgdata\train\1',loadsize,finesize,flip=1)
    datasetB = DATASET(r'D:\workspace\dataset\pixiv_coloring\imgdata\train\4',loadsize,finesize,flip=1)
    loader_A = torch.utils.data.DataLoader(dataset=datasetA,
                                        batch_size=batchsize,
                                        shuffle=True,
                                        num_workers=2)
    loaderA = iter(loader_A)
    loader_B = torch.utils.data.DataLoader(dataset=datasetB,
                                        batch_size=batchsize,
                                        shuffle=True,
                                        num_workers=2)
    loaderB = iter(loader_B)
    ABPool = ImagePool(poolsize)
    BAPool = ImagePool(poolsize)

    valid_setA = DATASET(r'D:\workspace\dataset\pixiv_coloring\imgdata\val\1',finesize,finesize,flip=0)
    valid_setB = DATASET(r'D:\workspace\dataset\pixiv_coloring\imgdata\val\4',finesize,finesize,flip=0)
    loader_valid_A = torch.utils.data.DataLoader(dataset=valid_setA,
                                        batch_size=batchsize,
                                        shuffle=False,
                                        num_workers=2)
    #loader_valid_A = iter(loader_valid_A)
    loader_valid_B = torch.utils.data.DataLoader(dataset=valid_setB,
                                        batch_size=batchsize,
                                        shuffle=False,
                                        num_workers=2)
    #loader_valid_B = iter(loader_valid_B)

    ###########   MODEL   ###########
    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    nc = 3

    D_A = Discriminator(input_nc,ndf)
    D_B = Discriminator(output_nc,ndf)
    G_AB = Generator(input_nc, output_nc, ngf)
    G_BA = Generator(output_nc, input_nc, ngf)

    G_AB.apply(weights_init)
    G_BA.apply(weights_init)

    if(path_g_ab != ''):
        print('Warning! Loading pre-trained weights.')
        G_AB.load_state_dict(torch.load(path_g_ab))
        G_BA.load_state_dict(torch.load(path_g_ba))

    if(cuda):
        D_A.cuda()
        D_B.cuda()
        G_AB.cuda()
        G_BA.cuda()

    D_A.apply(weights_init)
    D_B.apply(weights_init)

    ###########   LOSS & OPTIMIZER   ##########
    criterionMSE = nn.L1Loss()
    if(loss_type == 'bce'):
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    # chain is used to update two generators simultaneously
    optimizerD_A = torch.optim.Adam(D_A.parameters(),lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
    optimizerD_B = torch.optim.Adam(D_B.parameters(),lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
    optimizerG = torch.optim.Adam(chain(G_AB.parameters(),G_BA.parameters()),lr=lr, betas=(beta1, 0.999))

    ###########   GLOBAL VARIABLES   ###########
    input_nc = input_nc
    output_nc = output_nc
    finesize = finesize

    real_A = torch.FloatTensor(batchsize, input_nc, finesize, finesize)
    AB = torch.FloatTensor(batchsize, input_nc, finesize, finesize)
    real_B = torch.FloatTensor(batchsize, output_nc, finesize, finesize)
    BA = torch.FloatTensor(batchsize, output_nc, finesize, finesize)
    label = torch.FloatTensor(batchsize)

    real_A = Variable(real_A)
    real_B = Variable(real_B)
    label = Variable(label)
    AB = Variable(AB)
    BA = Variable(BA)

    if(cuda):
        real_A = real_A.cuda()
        real_B = real_B.cuda()
        label = label.cuda()
        AB = AB.cuda()
        BA = BA.cuda()
        criterion.cuda()
        criterionMSE.cuda()

    real_label = 1
    fake_label = 0

    ###########   Testing    ###########
    def test(niter):
        for i, data in enumerate(loader_valid_A, 0):
            real_A.data.resize_(data.size()).copy_(data)
            AB = G_AB(real_A)
            ABA = G_BA(AB)
            vutils.save_image(AB.data,
            'out_imgs\\%03d_niter_%03d_AB.png' % (i, niter),
            normalize=True)
            vutils.save_image(ABA.data,
            'out_imgs\\%03d_niter_%03d_ABA.png' % (i, niter),
            normalize=True)
            Loggers['img_log'].image_summary(str(i)+'_AB', AB.cpu().data.numpy(), niter)
            Loggers['img_log'].image_summary(str(i)+'_ABA', ABA.cpu().data.numpy(), niter)

        for i, data in enumerate(loader_valid_B, 0):
            real_B.data.resize_(data.size()).copy_(data)
            BA = G_BA(real_B)
            BAB = G_AB(BA)
            vutils.save_image(BA.data,
            'out_imgs\\%03d_niter_%03d_BA.png' % (i, niter),
            normalize=True)
            vutils.save_image(BAB.data,
            'out_imgs\\%03d_niter_%03d_BAB.png' % (i, niter),
            normalize=True)
            Loggers['img_log'].image_summary(str(i)+'_BA', BA.cpu().data.numpy(), niter)
            Loggers['img_log'].image_summary(str(i)+'_BAB', BAB.cpu().data.numpy(), niter)
        
    ###########   Training   ###########
    D_A.train()
    D_B.train()
    G_AB.train()
    G_BA.train()
    for iteration in range(1,niter+1):
        ###########   data  ###########
        try:
            imgA = loaderA.next()
            imgB = loaderB.next()
        except StopIteration:
            loaderA, loaderB = iter(loader_A), iter(loader_B)
            imgA = loaderA.next()
            imgB = loaderB.next()

        real_A.data.resize_(imgA.size()).copy_(imgA)
        real_B.data.resize_(imgB.size()).copy_(imgB)

        ###########   fDx   ###########
        D_A.zero_grad()
        D_B.zero_grad()

        # train with real
        outA = D_A(real_A)
        outB = D_B(real_B)
        label.data.resize_(outA.size())
        label.data.fill_(real_label)
        l_A = criterion(outA, label)
        l_B = criterion(outB, label)
        errD_real = l_A + l_B
        errD_real.backward()

        # train with fake
        label.data.fill_(fake_label)

        AB_tmp = G_AB(real_A)
        AB.data.resize_(AB_tmp.data.size()).copy_(ABPool.Query(AB_tmp.cpu().data))
        BA_tmp = G_BA(real_B)
        BA.data.resize_(BA_tmp.data.size()).copy_(BAPool.Query(BA_tmp.cpu().data))
        
        out_BA = D_A(BA.detach())
        out_AB = D_B(AB.detach())

        l_BA = criterion(out_BA,label)
        l_AB = criterion(out_AB,label)

        errD_fake = l_BA + l_AB
        errD_fake.backward()

        errD = (errD_real + errD_fake)*0.5
        optimizerD_A.step()
        optimizerD_B.step()

        ########### fGx ###########
        G_AB.zero_grad()
        G_BA.zero_grad()
        label.data.fill_(real_label)

        AB = G_AB(real_A)
        ABA = G_BA(AB)

        BA = G_BA(real_B)
        BAB = G_AB(BA)

        out_BA = D_A(BA)
        out_AB = D_B(AB)

        l_BA = criterion(out_BA,label)
        l_AB = criterion(out_AB,label)

        # reconstruction loss (lambda)
        l_rec_ABA = criterionMSE(ABA, real_A) * 10
        l_rec_BAB = criterionMSE(BAB, real_B) * 10

        errGAN = l_BA + l_AB
        errMSE =  l_rec_ABA + l_rec_BAB
        errG = errGAN + errMSE
        errG.backward()

        optimizerG.step()

        ###########   Logging   ############
        if(iteration % log_step):
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_MSE: %.4f'
                    % (iteration, niter,
                        errD.data[0], errGAN.data[0], errMSE.data[0]))

            Loggers['Loss_D'].scalar_summary('loss', errD.data[0], iteration)
            Loggers['Loss_G'].scalar_summary('loss', errGAN.data[0], iteration)
            Loggers['Loss_MSE'].scalar_summary('loss', errMSE.data[0], iteration)

        ########## Visualize #########
        if(iteration % test_step == 0):
            test(iteration)

        if iteration % save_step == 0:
            torch.save(G_AB.state_dict(), '{}/G_AB_{}.pth'.format(ckp_dir, iteration))
            torch.save(G_BA.state_dict(), '{}/G_BA_{}.pth'.format(ckp_dir, iteration))
            torch.save(D_A.state_dict(), '{}/D_A_{}.pth'.format(ckp_dir, iteration))
            torch.save(D_B.state_dict(), '{}/D_B_{}.pth'.format(ckp_dir, iteration))

if __name__ == '__main__':
    main()