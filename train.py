#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,6,7,8"

from torch.autograd import Variable
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import open3d as o3d
from data import MVP
from model import IDAM, GNN
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
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
from utils.vis_utils import plot_single_pcd
from utils.train_utils import *

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
        R_ab.astype('float32'), translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
        euler_ab.astype('float32'), euler_ba.astype('float32')


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
                elif ind == len(varying_constant_epochs)-1 and epoch >= ep:
                    alpha = varying_constant[ind+1]
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

            #旋转前的点云
            out, loss, net_loss = net(incomplete_pointcloud1, complete_pointcloud1, alpha=alpha)

            train_loss_meter.update(net_loss.mean().item())
            net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
            optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(cfg_dict["exp_name"] + ' train [%d: %d/%d] reg: %s, loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
                             (epoch, i, len(train_loader.dataset) / args.batch_size, "n", args.loss, loss.mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha))

            #旋转后的点云
            out, loss, net_loss = net(incomplete_pointcloud2, complete_pointcloud2, alpha=alpha)

            train_loss_meter.update(net_loss.mean().item())
            net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
            optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(cfg_dict["exp_name"] + ' train [%d: %d/%d] reg: %s, loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
                             (epoch, i, len(train_loader.dataset) / args.batch_size, "y", args.loss,
                              loss.mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha))

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % cfg_dict["log_dir"], net, net_d=net_d)
            logging.info("Saving net...")

        if (epoch + 1) % args.epoch_interval_to_val == 0 or epoch == args.epochs_complt - 1:
            val(args, net, epoch, val_loss_meters, test_loader, best_epoch_losses, cfg_dict)

def val(args, net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, cfg_dict):
    idx_to_plot = [i for i in range(0, 41600, 100)]

    logging.info('Testing...')

    log_dir = os.path.dirname(args.load_model)
    # if args.save_vis:
    #     save_gt_path = os.path.join(log_dir, 'pics_train_reg', 'gt')
    #     save_partial_path = os.path.join(log_dir, 'pics_train_reg', 'partial')
    #     save_completion_path = os.path.join(log_dir, 'pics_train_reg', 'completion')
    #     os.makedirs(save_gt_path, exist_ok=True)
    #     os.makedirs(save_partial_path, exist_ok=True)
    #     os.makedirs(save_completion_path, exist_ok=True)

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
            #
            # if args.save_vis:
            #     for z in range(args.batch_size):
            #         idx = i * args.batch_size + z
            #         if idx in idx_to_plot:
            #             # 每75个样本保存一个
            #             pic = 'object_%d.png' % idx
            #             plot_single_pcd(result_dict['out2'][z].cpu().numpy(), os.path.join(save_completion_path, pic))
            #             plot_single_pcd(complete_pointcloud1[z].transpose(0,1).cpu().numpy(), os.path.join(save_gt_path, pic))
            #             plot_single_pcd(incomplete_pointcloud1[z].transpose(0,1).cpu().numpy(), os.path.join(save_partial_path, pic))

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (cfg_dict["log_dir"], loss_type), net)
                #如果当前验证集loss比最好的loss低则保存当前模型
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)
 
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
        fine_pred1 = result_dict1['out2'].cpu() #(B, 2048, 3)

        fine_pred2 = transform_point_cloud(fine_pred1.transpose(1,2), rotation_ab, translation_ab)

        src = fine_pred1.transpose(1,2).cuda() #(B, 3, 2048)
        target = fine_pred2.cuda() #(B, 3, 2048)
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

        print('loss:',loss)
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

def test(args, net_reg, net_complt, test_loader, boardio, textio):

    test_loss, test_cycle_loss, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch_reg(args, net_reg, net_complt, test_loader)
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
    opt = optim.Adam(net_reg.parameters(), lr=0.00005, weight_decay=0.001)
    scheduler = MultiStepLR(opt, milestones=[30], gamma=0.1)

    best_rot_mse = np.inf
    best_rot_rmse = np.inf
    best_rot_mae = np.inf
    best_trans_mse = np.inf
    best_trans_rmse = np.inf
    best_trans_mae = np.inf

    print("-------------training complete network...-------------")
    for epoch in range(args.epochs_complt):  # 进行多个epoch的训练
        train_one_epoch_complete(args, epoch, train_loader, test_loader, cfg_dict) #训练补全网络

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
    #     train_stats = train_one_epoch_registration(args, net_reg, net_complt, train_loader, opt)
    # 
    #     #Saving net
    #     textio.cprint('Saving net')
    #     torch.save(net_reg.state_dict(), 'checkpoints/%s/models/model.latest.t7' % args.exp_name_reg)
    # 
    #     test_stats = test_one_epoch_reg(args, net_reg, net_complt, test_loader)
    # 
    #     print('=====  EPOCH %d  =====' % (epoch+1))
    #     print('TRAIN, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE:%f, trans_RMSE: %f, trans_MAE: %f' % train_stats)
    #     print('TEST, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE:%f, trans_RMSE: %f, trans_MAE: %f' % test_stats)
    # 
    #     textio.cprint('===== EPOCH %d  =====' % (epoch+1))
    #     textio.cprint('TRAIN, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE:%f, trans_RMSE: %f, trans_MAE: %f' % train_stats)
    #     textio.cprint('TEST, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE:%f, trans_RMSE: %f, trans_MAE: %f' % test_stats)
    # 
    #     r_mse, r_rmse, r_mae, t_mse, t_rmse, t_mae = test_stats
    #     if((best_rot_mse > r_mse) or (best_trans_mse > t_mse)):
    #         best_rot_mse = r_mse
    #         best_rot_mae = r_mae
    #         best_rot_rmse = r_rmse
    #         best_trans_mse = t_mse
    #         best_trans_mae = t_mae
    #         best_trans_rmse = t_rmse
    #         if torch.cuda.device_count() > 1:
    #             torch.save(net_reg.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name_reg)
    #         else:
    #             torch.save(net_reg.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name_reg)
    #         # 更新最佳模型
    #     scheduler.step()


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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))


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
            model_path = 'checkpoints' + '/' + args.exp_name_reg + '/models/model.best_backup.0.000007.t7'
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            print("loaded regitration model: " + model_path)
            net_reg.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net_reg = nn.DataParallel(net_reg)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)q
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
