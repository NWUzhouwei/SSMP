# -*- coding: utf-8 -*-


from easydict import EasyDict as edict

import os


__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = '/home/sxz/下载/3D-SOC-Net-master/datasets/ShapeNet_20.json'
__C.DATASETS.PROTOTYPE_PATH                 = '/home/sxz/下载/3D-SOC-Net-master/prototype/shapenet_point.pkl'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/media/sxz/软件/datasets/shapenet/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.POINT_PATH            = '/media/sxz/软件/datasets/shapenet_point/%s/%s'+'.npy'
__C.DATASETS.PIX3D                          = edict()
__C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = '/home/sxz/下载/3D-SOC-Net-master/datasets/Pix3D_10.json'
__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/home/sxz/下载/pix3d/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = '/media/sxz/软件/pix3d/img/%s/%s.%s'
__C.DATASETS.PIX3D.POINT_PATH               = '/media/sxz/软件/pix3d/model/%s/%s/%s'


#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'ShapeNet'
# __C.DATASET.TEST_DATASET                    = 'Pix3D'


#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 42
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.BATCH_SIZE                        = 8
__C.CONST.N_VIEWS_RENDERING                 = 1        # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W                        = 128       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 128       # Dummy property for Pascal 3D
__C.CONST.NUM_WORKER                        = 4         # number of data workers
__C.CONST.WEIGHTS                           ='/home/sxz/下载/3D-SOC-Net-master/models/5. Resnet2point/output/checkpoints_stage2_2024-07-12_15-47-14/checkpoint-best-20'
#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True
__C.NETWORK.INITIAL_POINTS_NUM=2048
__C.NETWORK.GEN_POINTS=8192
#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN1                      = False
__C.TRAIN.RESUME_TRAIN2                      = True
__C.TRAIN.NUM_EPOCHS                        = 400
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 5e-5
__C.TRAIN.DECODER_LEARNING_RATE             = 5e-5

__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

# train
__C.TRAIN.lr1                                = 0.001
__C.TRAIN.lr2                                = 0.00001
__C.TRAIN.lr_decay                          = True
__C.TRAIN.lr_decay_interval                 = 10
__C.TRAIN.lr_decay_rate                     = 0.5
__C.TRAIN.lr_step_decay_epochs              = None
__C.TRAIN.lr_step_decay_rates               = None
__C.TRAIN.lr_clip1                           = 5.e-5
__C.TRAIN.lr_clip2                           = 1.e-6
__C.TRAIN.varying_constant                            = [0.01, 0.1, 0.5, 1]
__C.TRAIN.varying_constant_epochs                     =[5, 15, 30]

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
