import torch
from torch import optim, nn
from torch.autograd import Variable
import open3d as o3d
import utils.dataload
import utils.data_transforms
import utils.helpers
from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time
from test import test_net
from val import val_net

from models.encoder import Encoder
from models.decoder import SP_DecoderEigen3steps
from utils.average_meter import AverageMeter
from torch.utils.data import DataLoader
import  os
import utils
from torchvision import transforms
import utils.point_transforms
import logging
from logging import handlers


import threading

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from geomloss import SamplesLoss
from config import cfg
from tqdm import tqdm
from utils.cd import  calc_cd

import pickle

def pc_normalize(pc, radius):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m * radius
    return pc


def get_spherepoints(num_points, radius):
    ball_name = './balls/%d.xyz' % num_points
    ball = np.loadtxt(ball_name)
    ball = pc_normalize(ball, radius)
    return ball


def readprior(shapenet_dir):
    shapenet = open(shapenet_dir, 'rb')
    shapenet = pickle.load(shapenet)
    return shapenet


def getprior(sample_names, cfg):
    prior_dic = readprior(cfg.DATASETS.PROTOTYPE_PATH)
    lis = []
    for name in sample_names:
        tmp = prior_dic[name]
        lis.append(tmp)
    ret = torch.stack(lis,dim=0)
    ret = torch.FloatTensor(ret.float())
    return ret



def train(args):
        logger = logging.getLogger('my_logger')
        torch.backends.cudnn.benchmark = True
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

        ###数据增强

        strong_transforms = utils.data_transforms.Compose([
            utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
            utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.RandomFlip(),
            utils.data_transforms.RandomPermuteRGB(),
            utils.data_transforms.ToTensor(),
        ])

        val_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])


        ###加载数据
        train_dataset=utils.dataload.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
        val_dataset = utils.dataload.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset.get_dataset(utils.dataload.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, strong_transforms),
            batch_size=cfg.CONST.BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=True,
            drop_last=True)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset.get_dataset(
            utils.dataload.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
            batch_size=cfg.CONST.BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=False,
            drop_last=True)

        #初始化模型以及权重
        encoder = Encoder()
        decoder = SP_DecoderEigen3steps(args=cfg)



        # encoder.apply(utils.helpers.init_weights)
        # encoder.contextblock._initialize_weights()
        # decoder.apply(utils.helpers.init_weights)
        # attention.apply(utils.helpers.init_weights)
        lr=cfg.TRAIN.lr1

        if cfg.TRAIN.lr_decay:
            if cfg.TRAIN.lr_decay_interval and cfg.TRAIN.lr_step_decay_epochs:
                raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
            if  cfg.TRAIN.lr_step_decay_epochs:
                decay_epoch_list = [int(ep) for ep in cfg.TRAIN.lr_step_decay_epochs]
                decay_rate_list = [float(rt) for rt in cfg.TRAIN.lr_step_decay_rates]

        if cfg.TRAIN.varying_constant:
            varying_constant_epochs = [int(ep) for ep in cfg.TRAIN.varying_constant_epochs]
            varying_constant = [float(c) for c in cfg.TRAIN.varying_constant]
            assert len(varying_constant) == len(varying_constant_epochs) + 1


        #设置优化器
        if cfg.TRAIN.POLICY == 'adam':
            encoder_solver = torch.optim.AdamW(filter(lambda p: p.requires_grad, encoder.parameters()),
                                              lr=lr,
                                              betas=cfg.TRAIN.BETAS)
            decoder_solver = torch.optim.AdamW(decoder.parameters(),
                                              lr=lr,
                                              betas=cfg.TRAIN.BETAS)
            # attention_solver = torch.optim.Adam(attention.parameters(),
            #                                     lr=cfg.TRAIN.DECODER_LEARNING_RATE,
            #                                     betas=cfg.TRAIN.BETAS)

        elif cfg.TRAIN.POLICY == 'sgd':
            encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                             momentum=cfg.TRAIN.MOMENTUM)
            decoder_solver = torch.optim.SGD(decoder.parameters(),
                                             lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                             momentum=cfg.TRAIN.MOMENTUM)
            # attention_solver = torch.optim.SGD(attention.parameters(),
            #                                    lr=cfg.TRAIN.DECODER_LEARNING_RATE,
            #                                    momentum=cfg.TRAIN.MOMENTUM)


        else:
            raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))




        if torch.cuda.is_available():
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            # attention = attention.cuda()

        #loss
        emd_loss_function = SamplesLoss()


        #  加载预训练模型
        init_epoch = 0
        best_cd = 100
        best_epoch = -1
        if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN1:
            logger.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
            init_epoch = checkpoint['epoch_idx']
            best_cd = checkpoint['best_cd']
            best_epoch = checkpoint['best_epoch']

            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            lr=0.5**(init_epoch//30)*lr
            logger.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                         (init_epoch, best_cd, best_epoch))
            print("cd:"+str(best_cd))

        # Summary writer for TensorBoard
        output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s')
        # 使用不包含冒号的日期时间格式
        now = dt.now()
        date_str = now.strftime('%Y-%m-%d')  # 使用不包含冒号的日期格式
        time_str = now.strftime('%H-%M-%S')  # 使用不包含冒号的时间格式
        # 组合日期和时间字符串
        timestamp = f'{date_str}_{time_str}'
        log_dir = output_dir % f'logs_{timestamp}'
        ckpt_dir = output_dir % f'checkpoints_{timestamp}'
        train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

        # sphere_points = get_spherepoints(2048, 0.5)
        # sphere_points = torch.FloatTensor(sphere_points).unsqueeze(0).repeat(cfg.CONST.BATCH_SIZE, 1, 1).cuda()  # B,3,2048
        # sphere_points = sphere_points.transpose(2, 1).contiguous()

        for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):
            epoch_start_time = time()

            # Batch average meterics
            batch_time = AverageMeter()
            data_time = AverageMeter()
            encoder_losses = AverageMeter()
            refiner_losses = AverageMeter()

            # switch models to training mode
            encoder.train()
            decoder.train()

            total_cd_l1 = 0
            total_cd_l2 = 0
            batch_size=cfg.CONST.BATCH_SIZE
            n_batches = len(train_data_loader)

            if cfg.TRAIN.lr_decay:
                if cfg.TRAIN.lr_decay_interval:
                    if epoch_idx > 0 and epoch_idx % cfg.TRAIN.lr_decay_interval == 0:
                        lr = lr * cfg.TRAIN.lr_decay_rate
                elif cfg.TRAIN.lr_step_decay_epochs:
                    if epoch_idx in decay_epoch_list:
                        lr = lr * decay_rate_list[decay_epoch_list.index(epoch_idx)]
                if cfg.TRAIN.lr_clip1:
                    lr = max(lr, cfg.TRAIN.lr_clip1)
                for param_group in encoder_solver.param_groups:
                    param_group['lr'] = lr


                for param_group in decoder_solver.param_groups:
                    param_group['lr'] = lr

            batch_end_time = time()
            with tqdm(train_data_loader) as t:
                for batch_idx, (taxonomy_names, sample_names, rendering_images,
                                ground_truth_points) in enumerate(t):
                    # Measure data time
                    data_time.update(time() - batch_end_time)
                    prior = getprior(taxonomy_names, cfg)

                    # Get data from data loader
                    rendering_images = utils.helpers.var_or_cuda(rendering_images)
                    ground_truth_points = utils.helpers.var_or_cuda(ground_truth_points)
                    prior = utils.helpers.var_or_cuda(prior)

                    prior=prior.permute(0,2,1)
                    prior=prior.view(-1,3,2048)

                    # 计算每个坐标轴的最小值和最大值

                    # train generator
                    encoder_solver.zero_grad()
                    decoder_solver.zero_grad()


                    image_features = encoder(rendering_images)
                    generated_points = decoder(prior, image_features)

                    net_loss, loss_t = calc_cd(generated_points.permute(0, 2, 1), ground_truth_points.to(torch.float32))
                    reg_loss = decoder.orthogonal_regularizer() * 100
                    net_loss = net_loss.mean()
                    encoder_loss = net_loss + reg_loss

                    # print("train1" + str(time() - batch_end_time))
                    encoder_loss.backward()

                    encoder_solver.step()
                    decoder_solver.step()
                    cd_l2_item = torch.sum(loss_t).item() / batch_size * 1e4
                    total_cd_l2 += cd_l2_item
                    cd_l1_item = net_loss.item() * 1e4
                    total_cd_l1 += cd_l1_item


                    # Append loss to average metrics
                    encoder_losses.update(encoder_loss.item())

                    # Append loss to TensorBoard
                    n_itr = epoch_idx * n_batches + batch_idx
                    train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
                    logger.info('[Epoch %d/%d] BatchLoss = %.3f (s)  n_itr=%f' %
                         (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS,  encoder_loss.item(), n_itr))
                    # Tick / tock
                    batch_time.update(time() - batch_end_time)
                    batch_end_time = time()


                    t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, args.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches))
                    t.set_postfix(loss='%s' % ['%.4f' % l for l in [encoder_loss, total_cd_l2 / n_batches]])
            # Adjust learning rate
            # encoder_lr_scheduler.step()
            # decoder_lr_scheduler.step()
            # attention_lr_scheduler.step()


            # Append epoch loss to TensorBoard
            train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
            # print(encoder_losses.avg)

            # Tick / tock
            epoch_end_time = time()
            logger.info('[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f ' %
                         (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, epoch_end_time - epoch_start_time, encoder_losses.avg))

            # Update Rendering Views
            # Validate the training models
            if epoch_idx >300:
                cd = val_net(cfg, epoch_idx + 1, val_data_loader, val_writer, encoder, decoder)
                logger.info('[Epoch %d/%d] valcd = %.3f (s) '  %
                         (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS,cd))
                print("cd:"+str(cd*100))
            elif epoch_idx<=100 and epoch_idx%20==0:
                cd = val_net(cfg, epoch_idx + 1, val_data_loader, val_writer, encoder, decoder)
                logger.info('[Epoch %d/%d] valcd = %.3f (s) ' %
                         (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS,cd))
                print("cd:" + str(cd*100))


            # Save weights to file
            if cd < best_cd:
                file_name = cfg.DATASET.TRAIN_DATASET+'-checkpoint-best'
                best_cd = cd
                best_epoch = epoch_idx


                output_path = ckpt_dir
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                checkpoint = {
                    'epoch_idx': epoch_idx,
                    'best_cd': best_cd,
                    'best_epoch': best_epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                }
                output_path = os.path.join(ckpt_dir, file_name)
                torch.save(checkpoint, output_path)
                logger.info('Saved checkpoint to %s ...' % output_path)

            if epoch_idx >100 and epoch_idx %40==0:
                file_name =  cfg.DATASET.TRAIN_DATASET+'-checkpoint-last'


                output_path = ckpt_dir
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                checkpoint = {
                    'epoch_idx': epoch_idx,
                    'best_cd': best_cd,
                    'best_epoch': best_epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                }
                output_path = os.path.join(ckpt_dir, file_name)
                torch.save(checkpoint, output_path)
                logger.info('Saved checkpoint to %s ...' % output_path)

        # Close SummaryWriter for TensorBoard
        train_writer.close()
        val_writer.close()



if __name__ == '__main__':
    train(cfg)