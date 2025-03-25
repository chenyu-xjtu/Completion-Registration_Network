from __future__ import print_function
import torch
import numpy as np
from scipy.spatial.transform import Rotation

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.module.state_dict(),
                    'D_state_dict': net_d.module.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.module.state_dict()}, path)


def generator_step(net_d, out2, net_loss, optimizer):
    set_requires_grad(net_d, False)
    d_fake = net_d(out2[:, 0:2048, :])
    errG_loss_batch = torch.mean((d_fake - 1) ** 2)
    total_gen_loss_batch = errG_loss_batch + net_loss * 200
    total_gen_loss_batch.backward(torch.ones(torch.cuda.device_count()).cuda(), retain_graph=True, )
    optimizer.step()
    return d_fake


def discriminator_step(net_d, gt, d_fake, optimizer_d):
    set_requires_grad(net_d, True)
    d_real = net_d(gt[:, 0:2048, :])
    d_loss_fake = torch.mean(d_fake ** 2)
    d_loss_real = torch.mean((d_real - 1) ** 2)
    errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
    total_dis_loss_batch = errD_loss_batch
    total_dis_loss_batch.backward(torch.ones(torch.cuda.device_count()).cuda())
    optimizer_d.step()

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

if __name__ == '__main__':
    print(torch.Tensor([1,2]).cuda())



