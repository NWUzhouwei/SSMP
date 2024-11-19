import json
import imageio
import joblib
import pyvista
import scipy
import scipy.ndimage as nd
import scipy.io as io
import matplotlib
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.utils import data
import torch
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision.utils as vutils
import shutil
import utils
import random
from config import cfg
import gc
import trimesh
from geomloss import SamplesLoss
import sys
import pickle


transform = transforms.Compose([
    transforms.ToTensor(),
])

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

class BlockWrite(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


class Dataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args, t):
        assert t=='train' or t=='val' or t=='test' or t=='sample'

        self.args = args
        self.t = t
        if t=='train':
            self.images, self.pcds = joblib.load(self.args.data_dir + "train.pkl")
        elif t=='val':
            self.images, self.pcds = joblib.load(self.args.data_dir + "val.pkl")
        elif t=='test':
            self.images, self.pcds = joblib.load(self.args.data_dir + "test.pkl")
        elif t=='sample':
            self.images, self.pcds = joblib.load(self.args.data_dir + "samples.pkl")

        self.cal_Size()

        self.mean_shapes = joblib.load(self.args.data_dir + "mean_shapes(SOM).pkl")

    def cal_Size(self):
        self.category_size = []
        for i in range(len(self.pcds)):
            self.category_size.append(len(self.pcds[i]))

        self.dataset_size = sum(self.category_size)

    def __getitem__(self, i):
        if self.t == 'test':
            img_id = i%24
            i = i // 24
            cur_category, i = self.getCurCategory(i)
            img = self.images[cur_category][i][img_id]
            img = transform(img)
            return [img, self.pcds[cur_category][i], self.mean_shapes[cur_category], cur_category]

        cur_category, i = self.getCurCategory(i)

        img_num = len(self.images[cur_category][i])
        img = self.images[cur_category][i][np.random.randint(0, img_num)]
        img = transform(img)

        return [img, self.pcds[cur_category][i], self.mean_shapes[cur_category], cur_category]

    def __len__(self):
        if self.t=='test':
            return self.dataset_size*24
        return self.dataset_size

    def getCurCategory(self, i):
        for index in range(len(self.category_size)):
            if self.category_size[index] > i:
                return index, i
            else:
                i -= self.category_size[index]
        return 0, i

    def switchToTrain(self):
        self.images = []
        self.pcds = []
        gc.collect()
        self.images, self.pcds = joblib.load(self.args.data_dir + "train.pkl")
        self.t = 'train'
        self.cal_Size()

    def switchToVal(self):
        self.images = []
        self.pcds = []
        gc.collect()
        self.images, self.pcds = joblib.load(self.args.data_dir + "val.pkl")
        self.t = 'val'
        self.cal_Size()

    def switchToTest(self):
        self.images = []
        self.pcds = []
        gc.collect()
        self.images, self.pcds = joblib.load(self.args.data_dir + "test.pkl")
        self.t = 'test'
        self.cal_Size()

    def switchToSamples(self):
        self.images = []
        self.pcds = []
        gc.collect()
        self.images, self.pcds = joblib.load(self.args.data_dir + "samples.pkl")
        self.t = 'sample'
        self.cal_Size()

def chamfer_distance_with_batch(p1, p2, type):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :param type: sum or mean
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    assert type == 'sum' or type == 'mean'

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)

    p1 = p1.transpose(1, 2)

    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))

    dist = torch.norm(dist, 2, dim=3)

    dist = torch.min(dist, dim=2)[0]

    if type == 'mean':
        dist = torch.mean(dist)
    elif type == 'sum':
        dist = torch.sum(dist)

    return dist


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def read_pickle(path, G, G_solver):
    try:

        files = os.listdir(path)

        with open(path + files[0], "rb") as f:
            G.load_state_dict(torch.load(f))
        with open(path + files[1], "rb") as f:
            G_solver.load_state_dict(torch.load(f))

        recent_iter = files[0].split('_')[-1].split('.')[0]

        return recent_iter

    except Exception as e:

        print("fail try read_pickle", e)


def save_new_pickle(path, iteration, G, G_solver):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)


def SavePoints(points, path, name):
    point_cloud = pyvista.PolyData(points)
    # point_cloud = pyvista.PolyData(chair.cpu().detach().numpy()[0])
    sphere = pyvista.Sphere(radius=0.025)
    pc = point_cloud.glyph(scale=False, geom=sphere)
    cpos, img = pc.plot(background='white', cmap='Reds', show_scalar_bar=True,
                    cpos=[(1.6, 0.6, -1.3),
                          (0, 0, 0),
                          (0, 1, 0)],
                    screenshot=True,
                    off_screen=True,
                   return_cpos=True)
    imageio.imsave(path+name+'.png', img)


def validate(args, iteration, G, val_dataLoader, G_solver):
    mean_cd_loss = 0
    mean_emd_loss = 0

    emd_loss_function = SamplesLoss()

    for batch_idx, (taxonomy_names, sample_names, rendering_images,
                    ground_truth_points) in enumerate(val_dataLoader):

        images = utils.helpers.var_or_cuda(rendering_images)
        gt = utils.helpers.var_or_cuda(ground_truth_points)
        G.eval()
        prior = getprior(taxonomy_names, cfg)
        prior = utils.helpers.var_or_cuda(prior)
        fake_models = G(images,prior)

        mean_cd_loss += torch.add(chamfer_distance_with_batch(fake_models.permute(0,2,1).contiguous(), gt, type='mean'),
                                  chamfer_distance_with_batch(gt, fake_models.permute(0,2,1).contiguous(), type='mean')).item()

        mean_emd_loss += torch.mean(emd_loss_function(fake_models.permute(0,2,1).contiguous(), gt)).item()

    mean_cd_loss = mean_cd_loss / len(val_dataLoader)
    mean_emd_loss = mean_emd_loss / len(val_dataLoader)
    print("cd:"+str(mean_cd_loss))

    print("emd:"+str(mean_emd_loss))
    # f = open("val.log", 'a')
    # f.write("%s %f %f\n" % (iteration, mean_cd_loss, mean_emd_loss))

    # best_save_path = args.output_dir + "/best/"
    #
    # if args.min_loss > mean_cd_loss:
    #     args.min_loss = mean_cd_loss
    #     if os.path.exists(best_save_path):
    #         shutil.rmtree(best_save_path)
    #     save_new_pickle(best_save_path, iteration, G, G_solver)


def save_test_img(path, iteration, G, test_dataLoader):
    if not os.path.exists(path):
        os.makedirs(path)

    for i, data in enumerate(test_dataLoader, 0):

        images, models, mean_shape = data[0], data[1], data[2]
        vutils.save_image(images[0],
                          path + "/iteration_" + str(iteration).zfill(4) + "_test_image_" + str(i).zfill(4) + ".png")
        SavePoints(models.numpy()[0], path, "/iteration_" + str(iteration).zfill(4) + "_test_model_" + str(i).zfill(4))

        images = var_or_cuda(images)
        models = var_or_cuda(models)

        fake_models = G(images)

        SavePoints(fake_models.cpu().detach().numpy()[0], path,
                   "/iteration_" + str(iteration).zfill(4) + "_generate_model_" + str(i).zfill(4))

def save_test_result(path, G, test_dataLoader):
    if not os.path.exists(path):
        os.makedirs(path)
    emd_loss_function = SamplesLoss(loss='sinkhorn', debias=False, p=1, blur=1e-3, scaling=0.6)

    mean_cd_loss = []
    mean_emd_loss = []
    category=0
    cd_loss=0
    emd_loss=0
    cnt=0
    total_cd=0
    total_emd=0
    total_cnt=0
    for i, data in enumerate(test_dataLoader, 0):

        images, models, mean_shape, cur_category = data[0], data[1], data[2], data[3]
        if cur_category[0]!=category:
            cd_loss = cd_loss/cnt
            emd_loss = emd_loss/cnt
            mean_cd_loss.append(cd_loss)
            mean_emd_loss.append(emd_loss)
            cd_loss=0
            emd_loss=0
            cnt=0
            category = cur_category[0]

        images = var_or_cuda(images)
        models = var_or_cuda(models)

        fake_models = G(images)

        cd = torch.add(chamfer_distance_with_batch(fake_models, models, type='mean'),
                               chamfer_distance_with_batch(models, fake_models, type='mean')).item()

        emd = torch.mean(emd_loss_function(fake_models, models)).item()
        cd_loss+=cd
        emd_loss+=emd
        cnt+=1
        total_cd+=cd
        total_emd+=emd
        total_cnt+=1

    cd_loss = cd_loss / cnt
    emd_loss = emd_loss / cnt
    mean_cd_loss.append(cd_loss)
    mean_emd_loss.append(emd_loss)

    total_cd /= total_cnt
    total_emd /= total_cnt
    mean_cd_loss.append(total_cd)
    mean_emd_loss.append(total_emd)
    print(total_cnt)

    joblib.dump([mean_cd_loss, mean_emd_loss], path + "test_results.pkl")

def save_point_cloud(point,name):
    # 确保输入 tensor 的形状为 (1, 3, 2048)
    save_dir = 'SSMP'
    os.makedirs(save_dir, exist_ok=True)
    # 将 tensor 转换为 NumPy 数组并调整形状为 (2048, 3)
    point_cloud = point.squeeze(0).permute(1,0).cpu().detach().numpy()  # 从 (1, 3, 2048) 到 (2048, 3)

        # 创建点云对象
    cloud = trimesh.PointCloud(point_cloud)

        # 确保保存目录存在


    # 保存为 .obj 文件
    save_path = os.path.join(save_dir, f'{name}.obj')
    cloud.export(save_path)

