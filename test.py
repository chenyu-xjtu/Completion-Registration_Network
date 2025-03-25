#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"

from torch.autograd import Variable
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import MVP
from model import IDAM, GNN
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import open3d as o3d
from matplotlib import pyplot as plt
from utils.train_utils import *
from utils.vis_utils import *
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

# Part of the code is referred from: https://github.com/floodsung/LearningToCompare_FSL

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name_reg):
        os.makedirs('checkpoints/' + args.exp_name_reg)
    if not os.path.exists('checkpoints/' + args.exp_name_reg + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name_reg + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name_reg + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name_reg + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name_reg + '/' + 'data.py.backup')


def pointcloud_transform(args, pointcloud1):
    anglex = np.random.uniform() * np.pi / args.factor
    angley = np.random.uniform() * np.pi / args.factor
    anglez = np.random.uniform() * np.pi / args.factor

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])
    R_ab = Rx.dot(Ry).dot(Rz)
    R_ba = R_ab.T
    translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                               np.random.uniform(-0.5, 0.5)])
    translation_ba = -R_ba.dot(translation_ab)

    pointcloud1 = pointcloud1.T  # incomplete_pointcloud1是未配准前的缺失点云

    rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
    pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

    euler_ab = np.asarray([anglez, angley, anglex])
    euler_ba = -euler_ab[::-1]

    pointcloud1 = np.random.permutation(pointcloud1.T).T
    pointcloud2 = np.random.permutation(pointcloud2.T).T
    # 打乱各个点

    return pointcloud1.astype('float32'), pointcloud2.astype('float32'), \
        R_ab.astype('float32'), translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype(
        'float32'), \
        euler_ab.astype('float32'), euler_ba.astype('float32')


def train_one_epoch_registration(args, net_reg, net_complt, train_loader, opt):
    net_reg.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    batch_size = args.batch_size  # 这一批有几个点云
    num_points = args.num_points

    for i, data in enumerate(tqdm(train_loader), 0):
        # if i % 100 != 0:
        #     continue
        label, incomplete_pointcloud1, incomplete_pointcloud2, complete_pointcloud1, complete_pointcloud2, \
            rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = data

        inputs1 = incomplete_pointcloud1.float().cuda()
        inputs2 = incomplete_pointcloud2.float().cuda()
        gt1 = complete_pointcloud1.float().cuda()
        gt2 = complete_pointcloud2.float().cuda()
        gt1 = gt1.transpose(2, 1).contiguous()  # (2048,3)
        gt2 = gt2.transpose(2, 1).contiguous()  # (2048,3)
        inputs = [inputs1, inputs2]
        gt = [gt1, gt2]

        # j=0为旋转前的点云, j=1为旋转后的点云
        result_dict1 = net_complt(inputs[0], gt[0], is_training=False)
        fine_pred1 = result_dict1['out2'].cpu()  # (B, 2048, 3)

        fine_pred2 = transform_point_cloud(fine_pred1.transpose(1, 2), rotation_ab, translation_ab)

        src = fine_pred1.transpose(1, 2).cuda()  # (B, 3, 2048)
        target = fine_pred2.cuda()  # (B, 3, 2048)
        # src = incomplete_pointcloud1.cuda()
        # target = incomplete_pointcloud2.cuda()

        src = src.detach()
        target = target.detach()

        rotation_ab = rotation_ab.cuda()
        # (32,3,3)
        translation_ab = translation_ab.cuda()
        # (32,3)
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        R_pred, t_pred, loss = net_reg(src, target, rotation_ab, translation_ab)

        R_list.append(rotation_ab.detach().cpu().numpy())
        t_list.append(translation_ab.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler_ab.numpy())

        print('loss:', loss)
        loss.backward()
        opt.step()

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return r_mse, r_rmse, r_mae, t_mse, t_rmse, t_mae


def test_one_epoch_reg(args, net_reg, net_complt, test_loader):
    idx_to_plot = [i for i in range(0, 41600, 100)]
    net_reg.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []


    log_dir = "checkpoints/" + args.exp_name_reg
    save_gt_path = os.path.join(log_dir, 'pics', 'gt')
    save_origin_path = os.path.join(log_dir, 'pics', 'origin')
    save_reg_path = os.path.join(log_dir, 'pics', 'reg')
    save_orireg_path = os.path.join(log_dir, 'pics', 'orireg')
    save_reggt_path = os.path.join(log_dir, 'pics', 'reggt')
    save_origt_path = os.path.join(log_dir, 'pics', 'origt')

    os.makedirs(save_gt_path, exist_ok=True)
    os.makedirs(save_origin_path, exist_ok=True)
    os.makedirs(save_reg_path, exist_ok=True)
    os.makedirs(save_orireg_path, exist_ok=True)
    os.makedirs(save_reggt_path, exist_ok=True)
    os.makedirs(save_origt_path, exist_ok=True)

    batch_size = args.batch_size

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader), 0):
            # if i % 100 != 0:
            #     continue
            label, incomplete_pointcloud1, incomplete_pointcloud2, complete_pointcloud1, complete_pointcloud2, \
                rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = data

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
            fine_pred1 = result_dict1['out2'].cpu()  # (B, 2048, 3)

            fine_pred2 = transform_point_cloud(fine_pred1.transpose(1, 2), rotation_ab, translation_ab)

            src = fine_pred1.transpose(1, 2).cuda()  # (B, 3, 2048)
            target = fine_pred2.cuda()  # (B, 3, 2048)

            src = src.detach()
            target = target.detach()

            rotation_ab = rotation_ab.cuda()
            # (32,3,3)
            translation_ab = translation_ab.cuda()
            # (32,3)
            rotation_ba = rotation_ba.cuda()
            translation_ba = translation_ba.cuda()

            batch_size = src.size(0)
            num_examples += batch_size

            R_pred, t_pred, *_ = net_reg(src, target)

            R_list.append(rotation_ab.detach().cpu().numpy())
            t_list.append(translation_ab.detach().cpu().numpy())
            R_pred_list.append(R_pred.detach().cpu().numpy())
            t_pred_list.append(t_pred.detach().cpu().numpy())
            euler_list.append(euler_ab.numpy())

            transformed_src = transform_point_cloud(src, R_pred, t_pred)

            for z in range(args.batch_size):
                idx = i * args.batch_size + z
                if idx in idx_to_plot:
                    # 每100个样本保存一个
                    pic = 'object_%d.png' % idx
                    plot_single_pcd(target[z].transpose(0, 1).cpu().numpy(),
                                    os.path.join(save_gt_path, pic))  # 画ground truth
                    plot_single_pcd(src[z].transpose(0, 1).cpu().numpy(), os.path.join(save_origin_path, pic))  # 画原点云
                    plot_single_pcd(transformed_src[z].transpose(0, 1).cpu().numpy(),
                                    os.path.join(save_reg_path, pic))  # 画预测配准后的点云
                    plot_single_pcd_two(src[z].transpose(0, 1).cpu().numpy(), transformed_src[z].transpose(0, 1).cpu().numpy(),
                                        os.path.join(save_orireg_path, pic))  # 画原点云和配准后的点云
                    plot_single_pcd_two(target[z].transpose(0, 1).cpu().numpy(),
                                        transformed_src[z].transpose(0, 1).cpu().numpy(),
                                        os.path.join(save_reggt_path, pic))  # 画配准后的点云和gt
                    plot_single_pcd_two(src[z].transpose(0, 1).cpu().numpy(),
                                        target[z].transpose(0, 1).cpu().numpy(),
                                        os.path.join(save_origt_path, pic))  # 画input:源点云和目标点云

        R = np.concatenate(R_list, axis=0)
        t = np.concatenate(t_list, axis=0)
        R_pred = np.concatenate(R_pred_list, axis=0)
        t_pred = np.concatenate(t_pred_list, axis=0)
        euler = np.concatenate(euler_list, axis=0)

        euler_pred = npmat2euler(R_pred)
        r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
        r_rmse = np.sqrt(r_mse)
        r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
        t_mse = np.mean((t - t_pred) ** 2)
        t_rmse = np.sqrt(t_mse)
        t_mae = np.mean(np.abs(t - t_pred))

        return r_mse, r_rmse, r_mae, t_mse, t_rmse, t_mae


def test(args, net_reg, test_loader, boardio, textio):
    print("-------------testing registration network...-------------")
    # 导入补全模型
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net_complt = torch.nn.DataParallel(model_module.PCN(args))
    net_complt.cuda()
    if hasattr(model_module, 'weights_init'):
        net_complt.module.apply(model_module.weights_init)
    if args.load_model:
        ckpt = torch.load(args.load_model)
        net_complt.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    test_stats = test_one_epoch_reg(args, net_reg, net_complt, test_loader)

    print('TEST, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE:%f, trans_RMSE: %f, trans_MAE: %f' % test_stats)

    textio.cprint(
        'TEST, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE:%f, trans_RMSE: %f, trans_MAE: %f' % test_stats)


def main():
    # name or flags - 选项字符串的名字或者列表，例如foo 或者 - f, --foo。
    # action - 命令行遇到参数时的动作，默认值是store。
    # store_const，表示赋值为const；
    # append，将遇到的值存储成列表，也就是如果参数重复则会保存多个值;
    # append_const，将参数规范中定义的一个值保存到一个列表；
    # count，存储遇到的次数；此外，也可以继承argparse.Action自定义参数解析；
    # nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号，当不指定值时对于Positional argument使用default，对于Optional argument使用const；或者是 * 号，表示0或多个参数；或者是 + 号表示1或多个参数。
    # const - action和nargs所需要的常量值。
    # default - 不指定参数时的默认值。
    # type - 命令行参数应该被转换成的类型。
    # choices - 参数可允许的值的一个容器。
    # required - 可选参数是否可以省略(仅针对可选参数)。
    # help - 参数的帮助信息，当指定为argparse.SUPPRESS时表示不显示该参数的帮助信息.
    # metavar - 在usage说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
    # dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线.
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('-c', '--config', help='path to config file', required=True)

    # 以下是设置随机数
    # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name_reg)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name_reg + '/run.log')
    textio.cprint(str(args))

    dataset = MVP(prefix="train", num_points=args.num_points, gaussian_noise=args.gaussian_noise,
                  unseen=args.unseen, factor=args.factor)
    dataset_test = MVP(prefix="test", num_points=args.num_points, gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=int(args.workers))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                              num_workers=int(args.workers))

    if args.model == 'dcp':
        ##### load model ####
        net_reg = IDAM(GNN(args.emb_dims), args).cuda()

        if torch.cuda.device_count() > 1:
            net_reg = nn.DataParallel(net_reg)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        if args.eval:
            # if args.model_path == 'null':
            #     model_path = 'checkpoints' + '/' + args.exp_name_reg + '/models/model.15.t7'
            # else:
            #     model_path = args.model_path
            model_path = 'checkpoints' + '/' + args.exp_name_reg + '/models/model.latest_backup0.000006.t7'
            print("loaded regitration model: " + model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net_reg.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net_reg = nn.DataParallel(net_reg)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    # pcn_cd_debug_2023-01-11T17:51:55文件夹为本次运行记录日志，里面保存了训练到终止的模型参数，损失最小的best模型参数
    cfg_dict = {'exp_name': exp_name, 'log_dir': log_dir}
    test(args, net_reg, test_loader, boardio, textio)

    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
