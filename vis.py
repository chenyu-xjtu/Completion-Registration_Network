import argparse
import os
import open3d as o3d
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import transform_point_cloud
from torch.autograd import Variable
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import DCP
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import open3d as o3d
from matplotlib import pyplot as plt
from utils.train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import sys
import argparse
from data import MVP
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5"

def vis(args):
    dataset = MVP(prefix="train", num_points=args.num_points, gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor)
    dataset_test = MVP(prefix="test", num_points=args.num_points, gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net_complt = torch.nn.DataParallel(model_module.PCN(args))
    net_complt.cuda()
    if hasattr(model_module, 'weights_init'):
        net_complt.module.apply(model_module.weights_init)
    if args.load_model:
        ckpt = torch.load(args.load_model)
        net_complt.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    net_complt.module.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            label, incomplete_pointcloud1, incomplete_pointcloud2, complete_pointcloud1, complete_pointcloud2, \
            R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba = data

            inputs1 = incomplete_pointcloud1.float().cuda()
            inputs2 = incomplete_pointcloud2.float().cuda()
            gt1 = complete_pointcloud1.float().cuda()
            gt2 = complete_pointcloud2.float().cuda()
            gt1 = gt1.transpose(2, 1).contiguous()  # (2048,3)
            gt2 = gt2.transpose(2, 1).contiguous()  # (2048,3)
            inputs = [inputs1, inputs2]
            gt = [gt1, gt2]

            # j=0为配准前的点云, j=1为配准后的点云
            result_dict1 = net_complt(inputs[0], gt[0], is_training=False)

            coarse1 = result_dict1["out1"]
            fine1 = result_dict1["out2"]

            result_dict2 = net_complt(inputs[1], gt[1], is_training=False)

            coarse2 = result_dict2["out1"]
            fine2 = result_dict2["out2"]

            coarse = [coarse1, coarse2]
            fine = [fine1, fine2]

            # visualize
            if (i == 0):
                if (curr_epoch_num == 0):
                    p1 = np.array(inputs[0][20].transpose(0, 1).cpu().detach())  # (2048,3)
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(p1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                    c1 = np.array(gt[0][20].cpu().detach())
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(c1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                    p1 = np.array(inputs[1][20].transpose(0, 1).cpu().detach())  # (2048,3)
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(p1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                    c1 = np.array(gt[1][20].cpu().detach())
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(c1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                dense_pred1 = np.array(fine[0][20].cpu().detach())
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
                points = np.asarray(point_cloud.points)
                colors = None
                ax = plt.axes(projection='3d')
                ax.view_init(90, -90)
                ax.axis("off")
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                plt.show()

                dense_pred1 = np.array(fine[1][20].cpu().detach())
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
                points = np.asarray(point_cloud.points)
                colors = None
                ax = plt.axes(projection='3d')
                ax.view_init(90, -90)
                ax.axis("off")
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--workers', type=int, default=0, metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--model_name', type=str, default='pcn', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--loss', type=str, default='cd', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--load_model', type=str, default='log/pcn_cd_debug_2023-01-11T17:51:55/network.pth', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    vis(args)
