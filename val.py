
import json
import numpy as np
import logging
import torch
import torch

import utils.dataload
import utils.data_transforms
import utils.helpers
from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time


from models.encoder import Encoder
from models.decoder import SP_DecoderEigen3steps

from utils.average_meter import AverageMeter
from torch.utils.data import DataLoader
import  os
import utils
import logging
from logging import handlers


import numpy as np
from geomloss import SamplesLoss
from config import cfg
from tqdm import tqdm

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

import matplotlib.pyplot as plt
import os
import pickle

# vis = open('./visshapenet.pkl','rb')
# lis = pickle.load((vis))
# print(lis)

# def showvoxel(vox, name, classid, trueid):
#     vox = vox.squeeze().__ge__(0.3)
#     vox = vox.detach().cpu().numpy()
#
#     fig1 = plt.figure(name, figsize=(20, 20))
#     ax1 = fig1.gca(projection='3d')
#     ax1.voxels(vox, edgecolor="#6e6e6e", facecolors='#F5F5F5')
#
#     ax1.grid(False)  # 默认True，风格线。
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_zticks([])
#     plt.axis('off')
#     if not os.path.exists('sn_our/' + classid + "/" + trueid + "/"):
#         os.makedirs('sn_our/' + classid + "/" + trueid + "/")
#
#     plt.savefig('sn_our/' + classid + "/" + trueid + "/" + name + '.png')
#     plt.clf()
#
#
# def showvoxel_back(vox, name, classid, trueid):
#     vox = vox.squeeze().__ge__(0.3)
#     vox = vox.detach().cpu().numpy()
#
#
#     fig1 = plt.figure(name, figsize=(20, 20))
#     ax1 = fig1.gca(projection='3d')
#     ax1.view_init(elev=30., azim=60)
#     ax1.voxels(vox, edgecolor="#6e6e6e", facecolors='#F5F5F5')
#
#     ax1.grid(False)  # 默认True，风格线。
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_zticks([])
#     plt.axis('off')
#     if not os.path.exists('sn_our/' + classid + "/" + trueid + "/"):
#         os.makedirs('sn_our/' + classid + "/" + trueid + "/")
#
#     plt.savefig('sn_our/' + classid + "/" + trueid + "/" + name + '_back.png')
#     plt.clf()

def show(point_cloud_1,point_cloud_2):
    fig = plt.figure()

    # 添加一个3d子图
    ax1 = fig.add_subplot(121, projection='3d')

    # 绘制第一个点云，设置点的大小为较小的值
    ax1.scatter(point_cloud_1[0,:, 0], point_cloud_1[0,:, 1], point_cloud_1[0,:, 2], s=1)

    # 设置子图的标题
    ax1.title.set_text('gt point ')

    # 添加第二个3d子图
    ax2 = fig.add_subplot(122, projection='3d')

    # 绘制第二个点云，设置点的大小为较小的值
    ax2.scatter(point_cloud_2[0,:, 0], point_cloud_2[0,:, 1], point_cloud_2[0,:, 2], s=1)

    # 设置子图的标题
    ax2.title.set_text('gen point')

    plt.show()

def val_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
            ):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.dataload.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.dataload.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=4,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True,
                                                       shuffle=False,
        drop_last=True)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder()
        decoder = SP_DecoderEigen3steps(arg=cfg)
        # attention = Attention(2048, 2, 0.1)

        encoder = encoder.cuda()
        decoder = decoder.cuda()
        # attention = attention.cuda()

        logging.info('Loading weights from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        # attention.load_state_dict(checkpoint['attention_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    criterion = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCELoss()
    emd_loss_function = SamplesLoss()


    # Testing loop
    n_samples = len(test_data_loader)

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()

    mean_cd_loss = []

    cd_loss = 0

    cnt = 0
    total_cd = 0
    total_emd = 0

    total_cnt = 0

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_points) in enumerate(test_data_loader):
        taxonomy_name = taxonomy_id

        with torch.no_grad():
            # Get data from data loader
            prior = getprior(taxonomy_name,cfg)

            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_points=utils.helpers.var_or_cuda(ground_truth_points)
            prior = utils.helpers.var_or_cuda(prior)

            # Test the encoder, decoder, refiner and merger
            prior = prior.permute(0, 2, 1)
            prior = prior.view(-1, 3, 2048)
            image_features = encoder(rendering_images)
            # prior_features = attention(image_features, prior)
            # image_features = image_features.unsqueeze(1)
            generated_points = decoder(prior, image_features)
            # generated_points = torch.mean(generated_points, dim=1).permute(0,2,1).type(torch.float64).contiguous()
            # print(image_features)


            encoder_loss = torch.mean(emd_loss_function(generated_points.permute(0,2,1).contiguous(),ground_truth_points.to(torch.float32)))

            # Append loss and accuracy to average metrics

            cd = torch.add(utils.helpers.chamfer_distance_with_batch(generated_points.permute(0,2,1), ground_truth_points.to(torch.float32), type='mean'),
                           utils.helpers.chamfer_distance_with_batch(ground_truth_points.to(torch.float32), generated_points.permute(0,2,1), type='mean')).item()

            # print(encoder_loss)
            cd_loss += cd
            cnt += 1
            total_cd += cd
            total_cnt += 1
    # show(ground_truth_points.cpu(), generated_points.permute(0, 2, 1).cpu())
    cd_loss = cd_loss / cnt

    mean_cd_loss.append(cd_loss)


    total_cd /= total_cnt

    mean_cd_loss.append(total_cd)

    # print("emd："+ str(emd_loss))
    return total_cd
