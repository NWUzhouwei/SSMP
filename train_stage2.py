# -*- coding: utf-8 -*-

import os
import logging

import torch
import torch.backends.cudnn
import torch.utils.data
import torch.nn as  nn
import utils.data_loaders_un
import utils.data_transforms
import utils.helpers

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time
import trimesh
from val import val_net
from models.encoder import Encoder
from models.decoder import SP_DecoderEigen3steps

from utils.average_meter import AverageMeter
from collections import OrderedDict
import pickle
import math
import numpy as  np
from utils.cd import  calc_cd
from geomloss import SamplesLoss
from config import cfg
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
import utils.point_transforms

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



def save_image(a,b,c,d,e,f):
    images=[a,b,c,d]
    image_names = [str(e)+'rendering_images',str(f)+'rendering_unlabel_slight', str(f)+'rendering_unlabel_strong1',
                   str(f)+'rendering_unlabel_strong2']
    save_dir = 'saved_images'
    os.makedirs(save_dir, exist_ok=True)

    # 遍历每张图像进行可视化和保存
    for i, img_tensor in enumerate(images):
        # 提取图像并调整形状
        image = img_tensor[0, 0].permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式


        plt.imshow(image)
        plt.axis('off')  # 关闭坐标轴

        # 保存图像为文件
        save_path = os.path.join(save_dir, f'{image_names[i]}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭当前图像以释放内存

        print(f'Saved {save_path}')

def save_point_cloud(a,b,c,d,e,f,g):
    # 确保输入 tensor 的形状为 (1, 3, 2048)
    point=[a,b,c,d,e]
    image_names = [str(f) + 'gp', str(g) + 'gup1',
                   str(g) + 'gup2',
                   str(g) + 'gut',
                   str(g)+'gufp']
    save_dir = 'saved_point'
    os.makedirs(save_dir, exist_ok=True)
    # 将 tensor 转换为 NumPy 数组并调整形状为 (2048, 3)
    for i, point_tensor in enumerate(point):
        point_cloud = point_tensor.squeeze(0).permute(1, 0).cpu().detach().numpy()  # 从 (1, 3, 2048) 到 (2048, 3)

        # 创建点云对象
        cloud = trimesh.PointCloud(point_cloud)

        # 确保保存目录存在


    # 保存为 .obj 文件
        save_path = os.path.join(save_dir, f'{image_names[i]}.obj')
        cloud.export(save_path)

    print(f'Saved point cloud as {save_path}')



momentum = 0.9996
tot_step = 30


@torch.no_grad()
def _update_teacher_model(now_step, encoder, decoder, encoder_tea, decoder_tea,
                          keep_rate=0.996):
    now_momentum = 1 - (1 - momentum) * (math.cos(math.pi * now_step / tot_step) + 1) / 2
    keep_rate = now_momentum

    student_encoder = encoder.state_dict()
    student_decoder = decoder.state_dict()

    new_teacher_encoder = OrderedDict()
    new_teacher_decoder = OrderedDict()

    for key, value in encoder_tea.state_dict().items():
        if key in student_encoder.keys():
            new_teacher_encoder[key] = (
                    student_encoder[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    for key, value in decoder_tea.state_dict().items():
        if key in student_decoder.keys():
            new_teacher_decoder[key] = (
                    student_decoder[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    encoder_tea.load_state_dict(new_teacher_encoder)
    decoder_tea.load_state_dict(new_teacher_decoder)

    return encoder_tea, decoder_tea


def finetune_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    logger = logging.getLogger('my_logger')
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W



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
    slight_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])


    # Set up data loader
    train_dataset_loader = utils.data_loaders_un.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders_un.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders_un.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, strong_transforms, slight_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True,
        drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders_un.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms, val_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=False,
        drop_last=True)

    # Set up networks
    encoder_stu = Encoder()
    decoder_stu = SP_DecoderEigen3steps(cfg)


    encoder_tea = Encoder()
    decoder_tea = SP_DecoderEigen3steps(cfg)


    lr = cfg.TRAIN.lr2

    if cfg.TRAIN.lr_decay:
        if cfg.TRAIN.lr_decay_interval and cfg.TRAIN.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if cfg.TRAIN.lr_step_decay_epochs:
            decay_epoch_list = [int(ep) for ep in cfg.TRAIN.lr_step_decay_epochs]
            decay_rate_list = [float(rt) for rt in cfg.TRAIN.lr_step_decay_rates]

    if cfg.TRAIN.varying_constant:
        varying_constant_epochs = [int(ep) for ep in cfg.TRAIN.varying_constant_epochs]
        varying_constant = [float(c) for c in cfg.TRAIN.varying_constant]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.AdamW(filter(lambda p: p.requires_grad, encoder_stu.parameters()),
                                          lr=lr,
                                          betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.AdamW(decoder_stu.parameters(),
                                          lr=lr,
                                          betas=cfg.TRAIN.BETAS)


    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder_stu.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder_stu.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)

    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))


    # Set up learning rate scheduler to decay learning rates dynamically
    # encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
    #                                                             milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
    #                                                             gamma=cfg.TRAIN.GAMMA)
    # decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
    #                                                             milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
    #                                                             gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder_stu = encoder_stu.cuda()
        decoder_stu = decoder_stu.cuda()

        encoder_tea = encoder_tea.cuda()
        decoder_tea = decoder_tea.cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    l2_loss = torch.nn.MSELoss()
    emd_loss_function = SamplesLoss()
    # Load pretrained model if exists
    init_epoch = 0
    best_cd= 10000
    best_epoch = -1

    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN2:
        logger.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        if torch.cuda.is_available():
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
        else:
            checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))
        init_epoch = 0
        best_CD = checkpoint['best_cd']

        best_epoch = checkpoint['best_epoch']
        # print(best_iou)
        encoder_tea.load_state_dict(checkpoint['encoder_state_dict'])
        decoder_tea.load_state_dict(checkpoint['decoder_state_dict'])
        encoder_stu.load_state_dict(checkpoint['encoder_state_dict'])
        decoder_stu.load_state_dict(checkpoint['decoder_state_dict'])


        logger.info('Recover complete. Current epoch #%d, Best CD = %.4f at epoch #%d.' %
                     (init_epoch, best_CD, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s')
    # 使用不包含冒号的日期时间格式
    now = dt.now()
    date_str = now.strftime('%Y-%m-%d')  # 使用不包含冒号的日期格式
    time_str = now.strftime('%H-%M-%S')  # 使用不包含冒号的时间格式
    # 组合日期和时间字符串
    timestamp = f'{date_str}_{time_str}'
    log_dir = output_dir % f'logs_stage2_{timestamp}'
    ckpt_dir = output_dir % f'checkpoints_stage2_{timestamp}'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    total_step = 0

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        encoder_losses = AverageMeter()
        refiner_losses = AverageMeter()

        # switch models to training mode
        encoder_stu.train()
        decoder_stu.train()

        encoder_tea.eval()
        decoder_tea.eval()

        total_cd_l2=0
        total_cd_l1=0
        n_batches = len(train_data_loader)
        batch_size=cfg.CONST.BATCH_SIZE

        # sphere_points = get_spherepoints(2048, 0.5)
        # sphere_points = torch.FloatTensor(sphere_points).unsqueeze(0).repeat(cfg.CONST.BATCH_SIZE, 1, 1).cuda()  # B,3,2048
        # sphere_points = sphere_points.transpose(2, 1).contiguous()

        if cfg.TRAIN.lr_decay:
            if cfg.TRAIN.lr_decay_interval:
                if epoch_idx > 0 and epoch_idx % cfg.TRAIN.lr_decay_interval == 0:
                    lr = lr * cfg.TRAIN.lr_decay_rate
            elif cfg.TRAIN.lr_step_decay_epochs:
                if epoch_idx in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch_idx)]
            if cfg.TRAIN.lr_clip2:
                lr = max(lr, cfg.TRAIN.lr_clip2)
            for param_group in encoder_solver.param_groups:
                param_group['lr'] = lr
            for param_group in decoder_solver.param_groups:
                param_group['lr'] = lr

        batch_end_time = time()
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_names, sample_names, rendering_images, unlabel_name,
                            rendering_unlabel_slight,rendering_unlabel_strong1 ,rendering_unlabel_strong2,
                            ground_truth_points) in enumerate(t):
                # Measure data time

                data_time.update(time() - batch_end_time)

                prior = getprior(taxonomy_names,cfg)
                un_prior = getprior(unlabel_name,cfg)

                prior = utils.helpers.var_or_cuda(prior)
                un_prior = utils.helpers.var_or_cuda(un_prior)
                prior = prior.permute(0, 2, 1)
                prior = prior.view(-1, 3, 2048)
                un_prior = un_prior.permute(0, 2, 1)
                un_prior = un_prior.view(-1, 3, 2048)

                # Get data from data loader
                rendering_images = utils.helpers.var_or_cuda(rendering_images)
                rendering_unlabel_slight = utils.helpers.var_or_cuda(rendering_unlabel_slight)
                rendering_unlabel_strong1 = utils.helpers.var_or_cuda(rendering_unlabel_strong1)
                rendering_unlabel_strong2 = utils.helpers.var_or_cuda(rendering_unlabel_strong2)
                ground_truth_points = utils.helpers.var_or_cuda(ground_truth_points)

                # save_image(rendering_images,rendering_unlabel_slight,rendering_unlabel_strong1,rendering_unlabel_strong2,sample_names,unlabel_names)

                image_features = encoder_stu(rendering_images)
                generated_points = decoder_stu(prior, image_features)

                net_loss, loss_t = calc_cd(generated_points.permute(0, 2, 1).to(torch.float32), ground_truth_points.to(torch.float32))
                reg_loss = decoder_stu.orthogonal_regularizer() * 100
                net_loss = net_loss.mean()
                loss_supervised = net_loss + reg_loss

                cd_l2_item = torch.sum(loss_t).item() / batch_size * 1e4
                total_cd_l2 += cd_l2_item
                total_cd_l1+= net_loss*1000
                # Train the encoder, decoder
                unlabel_features1 = encoder_stu(rendering_unlabel_strong1)
                generated_unlabel1 = decoder_stu(un_prior, unlabel_features1)

                unlabel_features2 = encoder_stu(rendering_unlabel_strong2)
                generated_unlabel2 = decoder_stu(un_prior, unlabel_features2)


                with torch.no_grad():
                    unlabel_features_tea = encoder_tea(rendering_unlabel_slight)
                    generated_unlabel_tea = decoder_tea(un_prior,unlabel_features_tea)

                    unlabel_features_tea_fp = encoder_tea(rendering_unlabel_slight)
                    generated_unlabel_tea_fp = decoder_tea(un_prior,nn.Dropout(0.5)(unlabel_features_tea_fp))

                    # score=torch.mean(emd_loss_function(generated_unlabel_tea, generated_unlabel)).item()

                # save_point_cloud(generated_points,generated_unlabel1,generated_unlabel2,generated_unlabel_tea,generated_unlabel_tea_fp,sample_names,unlabel_names)
                net_un_loss1, loss_t1 = calc_cd(generated_unlabel1.permute(0, 2, 1).to(torch.float32), generated_unlabel_tea.permute(0, 2, 1).to(torch.float32))
                net_un_loss1 = net_un_loss1.mean()
                loss_unsupervised1 = net_un_loss1

                net_un_loss2, loss_t = calc_cd(generated_unlabel2.permute(0, 2, 1).to(torch.float32), generated_unlabel_tea.permute(0, 2, 1).to(torch.float32))
                net_un_loss2 = net_un_loss2.mean()
                loss_unsupervised2 = net_un_loss2

                net_un_loss3, loss_t = calc_cd(generated_unlabel_tea_fp.permute(0, 2, 1).to(torch.float32), generated_unlabel_tea.permute(0, 2, 1).to(torch.float32))
                net_un_loss3 = net_un_loss3.mean()
                loss_unsupervised3 = net_un_loss3

                loss_unsupervised=0.25 * loss_unsupervised1 + 0.25 * loss_unsupervised2 + loss_unsupervised3 * 0.5
                # loss_unsupervised=loss_unsupervised1

                loss = (loss_supervised + loss_unsupervised)/2

                # print("train3" + str(time() - batch_end_time))
                # Gradient decent
                encoder_stu.zero_grad()
                decoder_stu.zero_grad()


                loss.backward()
                encoder_solver.step()
                decoder_solver.step()
                # Append loss to average metrics
                encoder_losses.update(loss.item())
                # Append loss to TensorBoard
                n_itr = epoch_idx * n_batches + batch_idx
                total_step += 1
                n_itr = epoch_idx * n_batches + batch_idx
                logger.info('[Epoch %d/%d] BatchLoss = %.3f (s)  n_itr=%f' %
                         (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS,  loss.item(), n_itr))
                # encoder_tea, decoder_tea = _update_teacher_model(epoch_idx, encoder_stu, decoder_stu,
                #                                                                 encoder_tea, decoder_tea, 0.99996)
                # Tick / tock
                batch_time.update(time() - batch_end_time)

                # print("optimizer" + str(time() - batch_end_time))

                batch_end_time = time()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [loss_supervised, loss_unsupervised,total_cd_l1/n_batches,total_cd_l2 / n_batches]])



        # Adjust learning rate
        # encoder_lr_scheduler.step()
        # decoder_lr_scheduler.step()

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        # train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        logger.info('[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f ' %
                     (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, epoch_end_time - epoch_start_time, encoder_losses.avg
                     ))

        # Update Rendering Views
        # Validate the training models


        cd = val_net(cfg, epoch_idx + 1, val_data_loader, val_writer, encoder_stu, decoder_stu)
        # print(cd*100)
        logger.info('[Epoch %d/%d] valcd = %.3f (s) ' %
                     (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, cd))

        # Save weights to file
        if cd < best_cd:
            file_name = cfg.DATASET.TRAIN_DATASET + '-checkpointstage2-best'
            best_cd = cd
            best_epoch = epoch_idx
            output_path = ckpt_dir
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            checkpoint = {
                'epoch_idx': epoch_idx,
                'best_cd': best_cd,
                'best_epoch': best_epoch,
                'encoder_state_dict': encoder_stu.state_dict(),
                'decoder_state_dict': decoder_stu.state_dict(),
            }
            output_path = os.path.join(ckpt_dir, file_name)
            torch.save(checkpoint, output_path)
            logger.info('Saved checkpoint to %s ...' % output_path)

        if epoch_idx >10 and epoch_idx %10==0:
            file_name = cfg.DATASET.TRAIN_DATASET + '-checkpointstage2-last'

            output_path = ckpt_dir
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            checkpoint = {
                'epoch_idx': epoch_idx,
                'best_cd': best_cd,
                'best_epoch': best_epoch,
                'encoder_state_dict': encoder_stu.state_dict(),
                'decoder_state_dict': decoder_stu.state_dict(),
            }
            output_path = os.path.join(ckpt_dir, file_name)
            torch.save(checkpoint, output_path)
            logger.info('Saved checkpoint to %s ...' % output_path)
    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    finetune_net(cfg)