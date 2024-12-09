import torch
import utils.dataload
import utils.data_transforms
import utils.helpers
import utils.util
from models.encoder import Encoder
from models.decoder import SP_DecoderEigen3steps
from torch.utils.data import DataLoader
import  os
import utils
import logging
import numpy as np
from geomloss import SamplesLoss
from config import cfg
from tqdm import tqdm
import  csv
import pickle

def pc_normalize(pc, radius):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m * radius
    return pc



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

def save_to_csv(taxonomy_data, filename="results.csv"):
    """
    将统计数据保存到CSV文件中
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header_written = False
        for taxonomy_id, samples in taxonomy_data.items():
            if not header_written:
                writer.writerow(['Category', 'Sample Name', 'CD Value'])
                header_written = True
            sorted_samples = sorted(samples.items(), key=lambda item: item[1])  # 按CD值排序
            for sample_name, cd in sorted_samples:
                writer.writerow([taxonomy_id, sample_name, cd])  # 写入类别、物体名称及对应的CD值

def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
            ):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    # taxonomies = []
    # with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
    #     taxonomies = json.loads(file.read())
    # taxonomies = {t['taxonomy_id']: t for t in taxonomies}

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
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True,
                                                       shuffle=False,
        drop_last=True)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder()
        decoder = SP_DecoderEigen3steps(args=cfg)

        encoder = encoder.cuda()
        decoder = decoder.cuda()


        logging.info('Loading weights from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])



    # Testing loop
    n_samples = len(test_data_loader)

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()

    mean_cd_loss = []
    cd_loss = 0
    cnt = 0
    total_cd = 0

    total_cnt = 0
    category_cd_totals = {}
    category_counts = {}
    taxonomy_sample_cds = {}


    for sample_idx, (taxonomy_ids, sample_name, rendering_images, ground_truth_points) in enumerate(tqdm(test_data_loader)):
        taxonomy_name = taxonomy_ids

        # print(str(sample_idx)+":")
        # print(taxonomy_name)
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



            # print(sample_idx)
            for i, taxonomy_id in enumerate(taxonomy_ids):

                taxonomy_name = taxonomy_id.item() if isinstance(taxonomy_id, torch.Tensor) else taxonomy_id
                sample_name_str = sample_name[i]
                generated_point = generated_points[i].unsqueeze(0)
                ground_truth_point = ground_truth_points[i].unsqueeze(0)
                save_name=sample_name_str+taxonomy_id
                utils.util.save_point_cloud(generated_point,save_name)
                cd = torch.add(
                    utils.helpers.chamfer_distance_with_batch(generated_point.permute(0, 2, 1), ground_truth_point.to(torch.float32),
                                                              type='mean'),
                    utils.helpers.chamfer_distance_with_batch(ground_truth_point.to(torch.float32), generated_point.permute(0, 2, 1),
                                                              type='mean')
                ).item()

                if taxonomy_name not in category_cd_totals:
                    category_cd_totals[taxonomy_name] = 0.0
                    category_counts[taxonomy_name] = 0
                    taxonomy_sample_cds[taxonomy_name] = {}
                # 累加CD值和计数
                category_cd_totals[taxonomy_name] += cd
                category_counts[taxonomy_name] += 1
                taxonomy_sample_cds[taxonomy_name][sample_name_str] = cd
            # Append loss and accuracy to average metrics
            #
            # cd = torch.add(utils.helpers.chamfer_distance_with_batch(generated_points.permute(0,2,1), ground_truth_points, type='mean'),
            #                utils.helpers.chamfer_distance_with_batch(ground_truth_points, generated_points.permute(0,2,1), type='mean')).item()


            # print(encoder_loss)
            cd_loss += cd
            cnt += 1
            total_cd += cd
            total_cnt += 1

    # 计算每个类别的平均CD
    category_average_cds = {category: category_cd_totals[category] / category_counts[category]
                            for category in category_cd_totals}

    # 打印每个类别的平均CD
    for category, average_cd in category_average_cds.items():
        print(
            f"Category {category}: Average CD = {category_average_cds[category]:.6f}")

     # 计算并打印总的平均CD
    total_cd = sum(category_cd_totals.values())
    total_count = sum(category_counts.values())

    overall_average_cd = total_cd / total_count

    print(f"Overall Average CD = {overall_average_cd:.6f}")

    # show(ground_truth_points.cpu(), generated_points.permute(0, 2, 1).cpu())
    cd_loss = cd_loss / cnt
    mean_cd_loss.append(cd_loss)
    total_cd /= total_cnt
    mean_cd_loss.append(total_cd)

    save_to_csv(taxonomy_sample_cds, "results.csv")

    return overall_average_cd

if __name__ == '__main__':
    test_net(cfg)