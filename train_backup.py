#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3,4,5"
from torch.autograd import Variable
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import MVP
from model import DCP
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
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


def train_one_epoch_complete(args, epoch, train_loader, test_loader, cfg_dict):
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.PCN(args))
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    net_d = None

    lr = args.lr_complt

    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    alpha = None
    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    for epoch in range(args.start_epoch, args.epochs_complt):
        train_loss_meter.reset()
        net.module.train()

        if args.varying_constant:
            for ind, ep in enumerate(varying_constant_epochs):
                if epoch < ep:
                    alpha = varying_constant[ind]
                    break
                elif ind == len(varying_constant_epochs) - 1 and epoch >= ep:
                    alpha = varying_constant[ind + 1]
                    break

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(tqdm(train_loader), 0):
            optimizer.zero_grad()

            label, incomplete_pointcloud1, incomplete_pointcloud2, complete_pointcloud1, complete_pointcloud2, \
                R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba = data

            # #旋转前的点云
            # out, loss, net_loss = net(incomplete_pointcloud1, complete_pointcloud1, alpha=alpha)
            #
            # train_loss_meter.update(net_loss.mean().item())
            # net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
            # optimizer.step()
            #
            # if i % args.step_interval_to_print == 0:
            #     logging.info(cfg_dict["exp_name"] + ' train [%d: %d/%d] reg: %s, loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
            #                  (epoch, i, len(train_loader.dataset) / args.batch_size, "n", args.loss, loss.mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha))

            # 旋转后的点云
            out, loss, net_loss = net(incomplete_pointcloud2, complete_pointcloud2, alpha=alpha)

            train_loss_meter.update(net_loss.mean().item())
            net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
            optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(cfg_dict[
                                 "exp_name"] + ' train [%d: %d/%d] reg: %s, loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
                             (epoch, i, len(train_loader.dataset) / args.batch_size, "y", args.loss,
                              loss.mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha))

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % cfg_dict["log_dir"], net, net_d=net_d)
            logging.info("Saving net...")

        # if epoch % args.epoch_interval_to_val == 0 or epoch == args.epochs_complt - 1:
        #     val(net, epoch, val_loss_meters, test_loader, best_epoch_losses, cfg_dict)


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, cfg_dict):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader_test)):

            label, incomplete_pointcloud1, incomplete_pointcloud2, complete_pointcloud1, complete_pointcloud2, \
                R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba = data

            inputs = incomplete_pointcloud2.float().cuda()
            gt = complete_pointcloud2.float().cuda()
            gt = gt.transpose(2, 1).contiguous()
            # result_dict = net(inputs, gt, is_training=False, mean_feature=mean_feature)
            result_dict = net(inputs, gt, is_training=False)
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())

            coarse = result_dict["out1"]
            fine = result_dict["out2"]

            # visualize
            if (i == 0):
                if (curr_epoch_num == 0):
                    p1 = np.array(inputs[10].transpose(0, 1).cpu().detach())  # (2048,3)
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(p1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                    c1 = np.array(gt[10].cpu().detach())
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(c1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                dense_pred1 = np.array(fine[10].cpu().detach())
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
                points = np.asarray(point_cloud.points)
                colors = None
                ax = plt.axes(projection='3d')
                ax.view_init(90, -90)
                ax.axis("off")
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                plt.show()

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (cfg_dict["log_dir"], loss_type), net)
                # 如果当前验证集loss比最好的loss低则保存当前模型
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)
        #     inputs1 = incomplete_pointcloud1.float().cuda()
        #     inputs2 = incomplete_pointcloud2.float().cuda()
        #     gt1 = complete_pointcloud1.float().cuda()
        #     gt2 = complete_pointcloud2.float().cuda()
        #     gt1 = gt1.transpose(2, 1).contiguous() #(2048,3)
        #     gt2 = gt2.transpose(2, 1).contiguous() #(2048,3)
        #     inputs = [inputs1, inputs2]
        #     gt = [gt1, gt2]
        #
        #     #j=0为配准前的点云, j=1为配准后的点云
        #     result_dict1 = net(inputs[0], gt[0], is_training=False)
        #     for k, v in val_loss_meters.items():
        #         #k是loss_type, v是AverageValueMeter(), result_dict[k]是loss值
        #         v.update(result_dict1[k].mean().item())
        #
        #     coarse1 = result_dict1["out1"]
        #     fine1 = result_dict1["out2"]
        #
        #     result_dict2 = net(inputs[1], gt[1], is_training=False)
        #     for k, v in val_loss_meters.items():
        #         #k是loss_type, v是AverageValueMeter(), result_dict[k]是loss值
        #         v.update(result_dict2[k].mean().item())
        #
        #     coarse2 = result_dict2["out1"]
        #     fine2 = result_dict2["out2"]
        #
        #     coarse = [coarse1, coarse2]
        #     fine = [fine1, fine2]
        #
        #     # visualize
        #     if (i < 20):
        #         if (curr_epoch_num == 0):
        #             p1 = np.array(inputs[0][10].transpose(0, 1).cpu().detach())  # (2048,3)
        #             point_cloud1 = o3d.geometry.PointCloud()
        #             point_cloud1.points = o3d.utility.Vector3dVector(p1)
        #             points = np.asarray(point_cloud1.points)
        #             colors = None
        #             ax = plt.axes(projection='3d')
        #             ax.view_init(90, -90)
        #             ax.axis("off")
        #             ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
        #             plt.show()
        #
        #             c1 = np.array(gt[0][10].cpu().detach())
        #             point_cloud2 = o3d.geometry.PointCloud()
        #             point_cloud2.points = o3d.utility.Vector3dVector(c1)
        #             points = np.asarray(point_cloud2.points)
        #             colors = None
        #             ax = plt.axes(projection='3d')
        #             ax.view_init(90, -90)
        #             ax.axis("off")
        #             ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
        #             plt.show()
        #
        #             p2 = np.array(inputs[1][10].transpose(0, 1).cpu().detach())  # (2048,3)
        #             point_cloud3 = o3d.geometry.PointCloud()
        #             point_cloud3.points = o3d.utility.Vector3dVector(p2)
        #             points = np.asarray(point_cloud3.points)
        #             colors = None
        #             ax = plt.axes(projection='3d')
        #             ax.view_init(90, -90)
        #             ax.axis("off")
        #             ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
        #             plt.show()
        #
        #             c2 = np.array(gt[1][10].cpu().detach())
        #             point_cloud4 = o3d.geometry.PointCloud()
        #             point_cloud4.points = o3d.utility.Vector3dVector(c2)
        #             points = np.asarray(point_cloud4.points)
        #             colors = None
        #             ax = plt.axes(projection='3d')
        #             ax.view_init(90, -90)
        #             ax.axis("off")
        #             ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
        #             plt.show()
        #
        #         dense_pred1 = np.array(fine[0][10].cpu().detach())
        #         point_cloud5 = o3d.geometry.PointCloud()
        #         point_cloud5.points = o3d.utility.Vector3dVector(dense_pred1)
        #         points = np.asarray(point_cloud5.points)
        #         colors = None
        #         ax = plt.axes(projection='3d')
        #         ax.view_init(90, -90)
        #         ax.axis("off")
        #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
        #         plt.show()
        #
        #         dense_pred2 = np.array(fine[1][10].cpu().detach())
        #         point_cloud6 = o3d.geometry.PointCloud()
        #         point_cloud6.points = o3d.utility.Vector3dVector(dense_pred2)
        #         points = np.asarray(point_cloud6.points)
        #         colors = None
        #         ax = plt.axes(projection='3d')
        #         ax.view_init(90, -90)
        #         ax.axis("off")
        #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
        #         plt.show()
        #
        # fmt = 'best_%s: %f [epoch %d]; '
        # best_log = ''
        # for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
        #     if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
        #             (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
        #         best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
        #         save_model('%s/best_%s_network.pth' % (cfg_dict["log_dir"], loss_type), net)
        #         #如果当前验证集loss比最好的loss低则保存当前模型
        #         logging.info('Best %s net saved!' % loss_type)
        #         best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
        #     else:
        #         best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)
        #
        # curr_log = ''
        # for loss_type, meter in val_loss_meters.items():
        #     curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)
        #
        # logging.info(curr_log)
        # logging.info(best_log)


def train_one_epoch_registration(args, net_reg, net_complt, train_loader, opt, epoch):
    net_reg.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []  # R真值
    translations_ab = []  # t真值
    rotations_ab_pred = []  # R计算值
    translations_ab_pred = []  # t计算值

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    batch_size = args.batch_size  # 这一批有几个点云
    num_points = args.num_points

    for i, data in enumerate(tqdm(train_loader), 0):
        if i % 100 != 0:
            continue
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
        result_dict2 = net_complt(inputs[1], gt[1], is_training=False)  # 缺失点云输入到生成器中，得到补全后的完整点云X',Y'

        # fine_pred1 = result_dict1['out2']
        # fine_pred2 = result_dict2['out2']
        fine_pred1 = result_dict1['out2']
        fine_pred2 = result_dict2['out2']

        src = fine_pred1.transpose(1, 2).cuda()
        target = fine_pred2.transpose(1, 2).cuda()
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
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net_reg(src, gt2.transpose(1,
                                                                                                                  2))  # 传入数据，开始训练

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        # torch.eye()生成单位矩阵
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)
        print('loss:', loss.item())
        # mse_loss(a,b) 均方误差，计算实际值a和预测值b之间平方差的平均值
        if args.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                        translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                           + translation_ba_pred) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

        mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

        if (epoch > 20):
            if (i < 500):
                if (epoch == 21):
                    p1 = np.array(inputs1[2].transpose(0, 1).cpu().detach())  # (2048,3)
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(p1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                    c1 = np.array(gt1[2].cpu().detach())
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(c1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                    p1 = np.array(inputs2[2].transpose(0, 1).cpu().detach())  # (2048,3)
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(p1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                    c1 = np.array(gt2[2].cpu().detach())
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(c1)
                    points = np.asarray(point_cloud.points)
                    colors = None
                    ax = plt.axes(projection='3d')
                    ax.view_init(90, -90)
                    ax.axis("off")
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                    plt.show()

                dense_pred1 = np.array(fine_pred1[2].cpu().detach())
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
                points = np.asarray(point_cloud.points)
                colors = None
                ax = plt.axes(projection='3d')
                ax.view_init(90, -90)
                ax.axis("off")
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                plt.show()

                dense_pred1 = np.array(fine_pred2[2].cpu().detach())
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
                points = np.asarray(point_cloud.points)
                colors = None
                ax = plt.axes(projection='3d')
                ax.view_init(90, -90)
                ax.axis("off")
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                plt.show()

                dense_pred1 = np.array(transformed_src[2].transpose(0, 1).cpu().detach())
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
                points = np.asarray(point_cloud.points)
                colors = None
                ax = plt.axes(projection='3d')
                ax.view_init(90, -90)
                ax.axis("off")
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
                plt.show()

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
        translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
        translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba


def test_one_epoch_reg(args, net_reg, net_complt, test_loader, epoch):
    net_reg.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    batch_size = args.batch_size

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader), 0):
            if i % 100 != 0:
                continue
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
            result_dict2 = net_complt(inputs[1], gt[1], is_training=False)  # 缺失点云输入到生成器中，得到补全后的虚假点云X',Y'

            # fine_pred1 = result_dict1['out2']
            # fine_pred2 = result_dict2['out2']
            fine_pred1 = result_dict1['out1']
            fine_pred2 = result_dict2['out2']

            src = fine_pred1.transpose(1, 2).cuda()
            target = fine_pred2.transpose(1, 2).cuda()
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
            num_examples += batch_size
            rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net_reg(src, gt2.transpose(1,
                                                                                                                      2))  # 传入数据，开始训练
            ## save rotation and translation
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.numpy())
            ##
            rotations_ba.append(rotation_ba.detach().cpu().numpy())
            translations_ba.append(translation_ba.detach().cpu().numpy())
            rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
            translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
            eulers_ba.append(euler_ba.numpy())

            transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

            transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

            ###########################
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
            loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                   + F.mse_loss(translation_ab_pred, translation_ab)
            if args.cycle:
                rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
                translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                            translation_ab_pred.view(batch_size, 3, 1)).view(batch_size,
                                                                                                             3)
                                               + translation_ba_pred) ** 2, dim=[0, 1])
                cycle_loss = rotation_loss + translation_loss

                loss = loss + cycle_loss * 0.1

            total_loss += loss.item() * batch_size

            if args.cycle:
                total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

            mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

            mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

            # if (i < 500):
            #     if (epoch == 0):
            #         p1 = np.array(inputs1[2].transpose(0, 1).cpu().detach())  # (2048,3)
            #         point_cloud = o3d.geometry.PointCloud()
            #         point_cloud.points = o3d.utility.Vector3dVector(p1)
            #         points = np.asarray(point_cloud.points)
            #         colors = None
            #         ax = plt.axes(projection='3d')
            #         ax.view_init(90, -90)
            #         ax.axis("off")
            #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
            #         plt.show()
            #
            #         c1 = np.array(gt1[2].cpu().detach())
            #         point_cloud = o3d.geometry.PointCloud()
            #         point_cloud.points = o3d.utility.Vector3dVector(c1)
            #         points = np.asarray(point_cloud.points)
            #         colors = None
            #         ax = plt.axes(projection='3d')
            #         ax.view_init(90, -90)
            #         ax.axis("off")
            #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
            #         plt.show()
            #
            #         p1 = np.array(inputs2[2].transpose(0, 1).cpu().detach())  # (2048,3)
            #         point_cloud = o3d.geometry.PointCloud()
            #         point_cloud.points = o3d.utility.Vector3dVector(p1)
            #         points = np.asarray(point_cloud.points)
            #         colors = None
            #         ax = plt.axes(projection='3d')
            #         ax.view_init(90, -90)
            #         ax.axis("off")
            #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
            #         plt.show()
            #
            #         c1 = np.array(gt2[2].cpu().detach())
            #         point_cloud = o3d.geometry.PointCloud()
            #         point_cloud.points = o3d.utility.Vector3dVector(c1)
            #         points = np.asarray(point_cloud.points)
            #         colors = None
            #         ax = plt.axes(projection='3d')
            #         ax.view_init(90, -90)
            #         ax.axis("off")
            #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
            #         plt.show()
            #
            #     dense_pred1 = np.array(fine_pred1[2].cpu().detach())
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
            #     points = np.asarray(point_cloud.points)
            #     colors = None
            #     ax = plt.axes(projection='3d')
            #     ax.view_init(90, -90)
            #     ax.axis("off")
            #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
            #     plt.show()
            #
            #     dense_pred1 = np.array(fine_pred2[2].cpu().detach())
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
            #     points = np.asarray(point_cloud.points)
            #     colors = None
            #     ax = plt.axes(projection='3d')
            #     ax.view_init(90, -90)
            #     ax.axis("off")
            #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
            #     plt.show()
            #
            #     dense_pred1 = np.array(transformed_src[2].transpose(0,1).cpu().detach())
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(dense_pred1)
            #     points = np.asarray(point_cloud.points)
            #     colors = None
            #     ax = plt.axes(projection='3d')
            #     ax.view_init(90, -90)
            #     ax.axis("off")
            #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
            #     plt.show()

        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

        rotations_ba = np.concatenate(rotations_ba, axis=0)
        translations_ba = np.concatenate(translations_ba, axis=0)
        rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
        translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

        eulers_ab = np.concatenate(eulers_ab, axis=0)
        eulers_ba = np.concatenate(eulers_ba, axis=0)

        return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
               mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
               mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
            translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
            translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba


def test(args, net_reg, net_complt, test_loader, boardio, textio):
    test_loss, test_cycle_loss, \
        test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
        test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch_reg(args, net_reg, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)
    test_rmse_ba = np.sqrt(test_mse_ba)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

    test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
    test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)) ** 2)
    test_r_rmse_ba = np.sqrt(test_r_mse_ba)
    test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
    test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
    test_t_rmse_ba = np.sqrt(test_t_mse_ba)
    test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab,
                     test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    textio.cprint('B--------->A')
    textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                     test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))


def train(args, net_reg, train_loader, test_loader, boardio, textio, cfg_dict):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net_reg.parameters(), lr=args.lr_reg * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net_reg.parameters(), lr=args.lr_reg, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    best_test_mse_ba = np.inf
    best_test_rmse_ba = np.inf
    best_test_mae_ba = np.inf

    best_test_r_mse_ba = np.inf
    best_test_r_rmse_ba = np.inf
    best_test_r_mae_ba = np.inf
    best_test_t_mse_ba = np.inf
    best_test_t_rmse_ba = np.inf
    best_test_t_mae_ba = np.inf

    print("-------------training complete network...-------------")
    for epoch in range(args.epochs_complt):  # 进行多个epoch的训练
        train_one_epoch_complete(args, epoch, train_loader, test_loader, cfg_dict)  # 训练补全网络

    # print("-------------training registration network...-------------")
    # #导入补全模型
    # model_module = importlib.import_module('.%s' % args.model_name, 'models')
    # net_complt = torch.nn.DataParallel(model_module.PCN(args))
    # net_complt.cuda()
    # if hasattr(model_module, 'weights_init'):
    #     net_complt.module.apply(model_module.weights_init)
    # if args.load_model:
    #     ckpt = torch.load(args.load_model)
    #     net_complt.module.load_state_dict(ckpt['net_state_dict'])
    #     logging.info("%s's previous weights loaded." % args.model_name)
    #
    # for epoch in range(args.epochs_reg):
    #     scheduler.step()
    #     train_loss, train_cycle_loss, \
    #     train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
    #     train_rotations_ab_pred, \
    #     train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
    #     train_translations_ba_pred, train_eulers_ab, train_eulers_ba = train_one_epoch_registration(args, net_reg, net_complt, train_loader, opt, epoch)
    #
    #     test_loss, test_cycle_loss, \
    #     test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    #     test_rotations_ab_pred, \
    #     test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    #     test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch_reg(args, net_reg, net_complt, test_loader,epoch)
    #
    #     train_rmse_ab = np.sqrt(train_mse_ab)
    #     test_rmse_ab = np.sqrt(test_mse_ab)
    #
    #     train_rmse_ba = np.sqrt(train_mse_ba)
    #     test_rmse_ba = np.sqrt(test_mse_ba)
    #
    #     train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
    #     train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
    #     train_r_rmse_ab = np.sqrt(train_r_mse_ab)
    #     train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
    #     train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
    #     train_t_rmse_ab = np.sqrt(train_t_mse_ab)
    #     train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))
    #
    #     train_rotations_ba_pred_euler = npmat2euler(train_rotations_ba_pred, 'xyz')
    #     train_r_mse_ba = np.mean((train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)) ** 2)
    #     train_r_rmse_ba = np.sqrt(train_r_mse_ba)
    #     train_r_mae_ba = np.mean(np.abs(train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)))
    #     train_t_mse_ba = np.mean((train_translations_ba - train_translations_ba_pred) ** 2)
    #     train_t_rmse_ba = np.sqrt(train_t_mse_ba)
    #     train_t_mae_ba = np.mean(np.abs(train_translations_ba - train_translations_ba_pred))
    #
    #     test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    #     test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    #     test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    #     test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    #     test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    #     test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    #     test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))
    #
    #     test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
    #     test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)) ** 2)
    #     test_r_rmse_ba = np.sqrt(test_r_mse_ba)
    #     test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
    #     test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
    #     test_t_rmse_ba = np.sqrt(test_t_mse_ba)
    #     test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))
    #
    #     if best_test_loss >= test_loss:
    #         best_test_loss = test_loss
    #         best_test_cycle_loss = test_cycle_loss
    #
    #         best_test_mse_ab = test_mse_ab
    #         best_test_rmse_ab = test_rmse_ab
    #         best_test_mae_ab = test_mae_ab
    #
    #         best_test_r_mse_ab = test_r_mse_ab
    #         best_test_r_rmse_ab = test_r_rmse_ab
    #         best_test_r_mae_ab = test_r_mae_ab
    #
    #         best_test_t_mse_ab = test_t_mse_ab
    #         best_test_t_rmse_ab = test_t_rmse_ab
    #         best_test_t_mae_ab = test_t_mae_ab
    #
    #         best_test_mse_ba = test_mse_ba
    #         best_test_rmse_ba = test_rmse_ba
    #         best_test_mae_ba = test_mae_ba
    #
    #         best_test_r_mse_ba = test_r_mse_ba
    #         best_test_r_rmse_ba = test_r_rmse_ba
    #         best_test_r_mae_ba = test_r_mae_ba
    #
    #         best_test_t_mse_ba = test_t_mse_ba
    #         best_test_t_rmse_ba = test_t_rmse_ba
    #         best_test_t_mae_ba = test_t_mae_ba
    #
    #         if torch.cuda.device_count() > 1:
    #             torch.save(net_reg.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name_reg)
    #         else:
    #             torch.save(net_reg.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name_reg)
    #         #更新最佳模型
    #     textio.cprint('==TRAIN==')
    #     textio.cprint('A--------->B')
    #     textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss:, %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
    #                   'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
    #                   % (epoch, train_loss, train_cycle_loss, train_mse_ab, train_rmse_ab, train_mae_ab, train_r_mse_ab,
    #                      train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))
    #     textio.cprint('B--------->A')
    #     textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
    #                   'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
    #                   % (epoch, train_loss, train_mse_ba, train_rmse_ba, train_mae_ba, train_r_mse_ba, train_r_rmse_ba,
    #                      train_r_mae_ba, train_t_mse_ba, train_t_rmse_ba, train_t_mae_ba))
    #
    #     textio.cprint('==TEST==')
    #     textio.cprint('A--------->B')
    #     textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
    #                   'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
    #                   % (epoch, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab, test_r_mse_ab,
    #                      test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    #     textio.cprint('B--------->A')
    #     textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
    #                   'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
    #                   % (epoch, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
    #                      test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))
    #
    #     textio.cprint('==BEST TEST==')
    #     textio.cprint('A--------->B')
    #     textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
    #                   'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
    #                   % (epoch, best_test_loss, best_test_cycle_loss, best_test_mse_ab, best_test_rmse_ab,
    #                      best_test_mae_ab, best_test_r_mse_ab, best_test_r_rmse_ab,
    #                      best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))
    #     textio.cprint('B--------->A')
    #     textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
    #                   'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
    #                   % (epoch, best_test_loss, best_test_mse_ba, best_test_rmse_ba, best_test_mae_ba,
    #                      best_test_r_mse_ba, best_test_r_rmse_ba,
    #                      best_test_r_mae_ba, best_test_t_mse_ba, best_test_t_rmse_ba, best_test_t_mae_ba))
    #
    #     # boardio.add_scalar('A->B/train/loss', train_loss, epoch)
    #     # boardio.add_scalar('A->B/train/MSE', train_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/train/RMSE', train_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/train/MAE', train_mae_ab, epoch)
    #     # boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae_ab, epoch)
    #     # boardio.add_scalar('A->B/train/translation/MSE', train_t_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/train/translation/MAE', train_t_mae_ab, epoch)
    #     #
    #     # boardio.add_scalar('B->A/train/loss', train_loss, epoch)
    #     # boardio.add_scalar('B->A/train/MSE', train_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/train/RMSE', train_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/train/MAE', train_mae_ba, epoch)
    #     # boardio.add_scalar('B->A/train/rotation/MSE', train_r_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/train/rotation/RMSE', train_r_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/train/rotation/MAE', train_r_mae_ba, epoch)
    #     # boardio.add_scalar('B->A/train/translation/MSE', train_t_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/train/translation/RMSE', train_t_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/train/translation/MAE', train_t_mae_ba, epoch)
    #     #
    #     # ############TEST
    #     # boardio.add_scalar('A->B/test/loss', test_loss, epoch)
    #     # boardio.add_scalar('A->B/test/MSE', test_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/test/RMSE', test_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/test/MAE', test_mae_ab, epoch)
    #     # boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae_ab, epoch)
    #     # boardio.add_scalar('A->B/test/translation/MSE', test_t_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)
    #     #
    #     # boardio.add_scalar('B->A/test/loss', test_loss, epoch)
    #     # boardio.add_scalar('B->A/test/MSE', test_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/test/RMSE', test_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/test/MAE', test_mae_ba, epoch)
    #     # boardio.add_scalar('B->A/test/rotation/MSE', test_r_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/test/rotation/RMSE', test_r_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/test/rotation/MAE', test_r_mae_ba, epoch)
    #     # boardio.add_scalar('B->A/test/translation/MSE', test_t_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/test/translation/RMSE', test_t_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/test/translation/MAE', test_t_mae_ba, epoch)
    #     #
    #     # ############BEST TEST
    #     # boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
    #     # boardio.add_scalar('A->B/best_test/MSE', best_test_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/best_test/RMSE', best_test_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/best_test/MAE', best_test_mae_ab, epoch)
    #     # boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
    #     # boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
    #     # boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
    #     # boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)
    #     #
    #     # boardio.add_scalar('B->A/best_test/loss', best_test_loss, epoch)
    #     # boardio.add_scalar('B->A/best_test/MSE', best_test_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/best_test/RMSE', best_test_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/best_test/MAE', best_test_mae_ba, epoch)
    #     # boardio.add_scalar('B->A/best_test/rotation/MSE', best_test_r_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/best_test/rotation/RMSE', best_test_r_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/best_test/rotation/MAE', best_test_r_mae_ba, epoch)
    #     # boardio.add_scalar('B->A/best_test/translation/MSE', best_test_t_mse_ba, epoch)
    #     # boardio.add_scalar('B->A/best_test/translation/RMSE', best_test_t_rmse_ba, epoch)
    #     # boardio.add_scalar('B->A/best_test/translation/MAE', best_test_t_mae_ba, epoch)
    #
    #     if torch.cuda.device_count() > 1:
    #         torch.save(net_reg.module.state_dict(), 'checkpoints/%s/models/model.15.t7' % args.exp_name_reg)
    #     else:
    #         torch.save(net_reg.state_dict(), 'checkpoints/%s/models/model.15.t7' % args.exp_name_reg)
    #     #更新最新模型
    #     gc.collect()
    #


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
        net_reg = DCP(args).cuda()
        if args.eval:
            # if args.model_path == 'null':
            #     model_path = 'checkpoints' + '/' + args.exp_name_reg + '/models/model.15.t7'
            # else:
            #     model_path = args.model_path
            model_path = 'checkpoints' + '/' + args.exp_name_reg + '/models/model.15.t7'
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
    train(args, net_reg, train_loader, test_loader, boardio, textio, cfg_dict)

    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
