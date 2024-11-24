#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

import logging
import matplotlib
import numpy as np
import os
import sys
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
# from pprint import pprint
import pprint
from config import cfg
from train_stage1 import train
from test import test_net
from train_stage2 import finetune_net
import torch
import random
from datetime import datetime as dt
def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--finetune',dest='finetune',help='Finetune Net',action='store_true')
    parser.add_argument('--datatype', dest='datatype', help='train dataset',  default=cfg.DATASET.TRAIN_DATASET)
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHS, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def set_seed(seeds):
    random.seed(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)


def main():
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s')
    now = dt.now()
    date_str = now.strftime('%Y-%m-%d')  # 使用不包含冒号的日期格式
    time_str = now.strftime('%H-%M-%S')  # 使用不包含冒号的时间格式
    timestamp = f'{date_str}_{time_str}'
    log_dir = output_dir % f'logs_{timestamp}'

    file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 获取日志记录器并添加文件处理器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # Get args from command line
    args = get_args_from_command_line()
    set_seed(cfg.CONST.RNG_SEED)

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)

    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHS = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True
    if args.datatype is not None:
        if args.datatype=="Pix3D":
            cfg.DATASET.TRAIN_DATASET = args.datatype
            cfg.DATASET.TEST_DATASET =args.datatype
            cfg.DATASETS.PROTOTYPE_PATH='prototype/pix3d_point.pkl'
        elif args.datatype=="ShapeNet":
            cfg.DATASET.TRAIN_DATASET = args.datatype
            cfg.DATASET.TEST_DATASET = args.datatype
            cfg.DATASETS.PROTOTYPE_PATH = 'prototype/shapenet_point.pkl'
    # Print config
    formatted_cfg = pprint.pformat(cfg)
    logger.info(f'Configuration:\n{formatted_cfg}')
    # print('Use config:')
    # pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if not args.test:
        if not args.finetune:
            train(cfg)
            logger.info('train first stage')
        else:
            logger.info('train finetune stage')
            finetune_net(cfg)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            test_net(cfg)
        else:
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)


if __name__ == '__main__':
    main()
