from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from utils.model_utils import gen_grid_up, calc_emd, calc_cd


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    # x(B,3,2048)
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    # idx(B,2048,20) 将输入点云数据通过在三维/特征空间上knn，idx记录了每个点在三维/特征空间上的k个最近点

    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    #idx_base(B,1,1)
    # tensor([[[   0]],
    #     [[2048]],
    #     [[4096]],
    #     [[6144]],
    #     [[8192]],
    #     [[10240]],
    #     [[12288]],
    #     [[14336]],
    #     [[16384]],
    #     [[18432]]], device='cuda:2')    #标识了该batch中每个点云（1024个点）的起始点号
    idx = idx + idx_base  #（B,2048,20）+（B,1,1）广播机制  #因为idx中存储的是元素在该点云2048个点中的索引，而idx_base记录了该batch中不同点云的首个点的点号，所以idx+idx_base的结果就可以准确对应索引到该batch中所有点的某个点。

    idx = idx.view(-1) #化为一维，（1310720），该batch中所有点的点号

    _, num_dims, _ = x.size()
    #x(B,3,2048)
    x = x.transpose(2,1).contiguous()
    #x(B,2048,3)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    # featrue(B*2048,3)
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = feature.view(batch_size, num_points, k, num_dims)
    # feature(B,2048,20,3)
    # 将k个邻近点的信息加入feature中

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # x(B,2048,20,3)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    # 将x原输入信息并入feature
    return feature #（B,6,2048,20)


def knn(x, k):
    #计算输入点云每对点之间的欧式距离，取离该点最近的k个点
    #以三维空间中两个点之间的距离为例，则为(x1-x2)方+(y1-y2)方+(z1-z2)方 = x1方+y1方+z1方+x2方+y2方+z2方-2(x1x2+y1y2+z1z2)

    #x(10,3,1024) batchsize, dim, pointnum

    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x) #matmul乘法
    #inner(10,1024,1024)，即-2(x1x2+y1y2+z1z2)

    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    # xx(10,1,1024)，即[x1方+y1方+z1方 , x2方+y2方+z2方, ... xn方+yn方+zn方] (横着的）
    # xx.transpose(2, 1)， 维度(10,1024,1) 即[[x1方+y1方+z1方] , [x2方+y2方+z2方], ... [xn方+yn方+zn方]]（竖着的）

    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous() #每对点之间的欧式距离
    #pairwise_distance(10,1024,1024)
    # 这里xx和xx.transpose利用广播机制，xx + xx.transpose(2, 1)即x1方+y1方+z1方+x2方+y2方+z2方

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    #topk()求tensor中某个dim的前k大或者前k小的值以及对应的index。
    #idx(10,1024,20)
    return idx # 1024个点，每个点有20个最近的点

class DGCNN(nn.Module):
    #x=f({h(xi,xj)}) （实际上这里用的EdgeConv只是二维卷积，没有h(xi, xj-xi))
    def __init__(self, emb_dims=256):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x) #①从点云数据获得图特征
        # x（B,6,2048,20)
        x = F.relu(self.bn1(self.conv1(x))) #②基于上面的图进行卷积
        # x（B,64,2048,20)
        x1 = x.max(dim=-1, keepdim=True)[0]
        # x1（B,64,2048,1)
        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]
        # x2（B,64,2048,1)
        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]
        # x3（B,128,2048,1)
        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]
        # x4（B,256,2048,1)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        # x（B,512,2048,1)
        # 把前四层特征都合并
        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        # x（B,512,2048)
        return x

class PCN_encoder(nn.Module):
    def __init__(self, output_size=1024):
        super(PCN_encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)
        self.dgcnn = DGCNN(emb_dims=256)

    def forward(self, x):
        batch_size, _, num_points = x.size()

        # x = F.relu(self.conv1(x))
        # x = self.conv2(x)
        x = self.dgcnn(x) #x(B,256,2048)

        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)


class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                3)).transpose(1,
                                                                                                              2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                      3)).transpose(1,
                                                                                                                    2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine


class Model(nn.Module):
    def __init__(self, args, num_coarse=1024):
        super(Model, self).__init__()

        self.num_coarse = num_coarse
        self.num_points = args.num_points
        self.train_loss = args.loss
        self.scale = self.num_points // num_coarse
        self.cat_feature_num = 2 + 3 + 1024

        self.encoder = PCN_encoder()
        self.decoder = PCN_decoder(num_coarse, self.num_points, self.scale, self.cat_feature_num)

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        feat = self.encoder(x)
        out1, out2 = self.decoder(feat)
        out1 = out1.transpose(1, 2).contiguous()
        out2 = out2.transpose(1, 2).contiguous()

        if is_training:
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                loss1, _ = calc_cd(out1, gt)
                loss2, _ = calc_cd(out2, gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss = loss1.mean() + loss2.mean() * alpha
            return out2, loss2, total_train_loss
        else:
            emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
            return {'out1': out1, 'out2': out2, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}