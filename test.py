from __future__ import print_function
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import time
# import math
from metrics import *
from tqdm import tqdm
from data_loader_matterport3d import Dataset
# from dataset_loader_stanford import Dataset
import cv2
# import supervision as L
# import spherical as S360
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
from model.spherical_fusion import spherical_fusion
# from model.spherical_fusion import *
# from ply import write_ply
# import csv
from util import *
import shutil
import torchvision.utils as vutils
# from pytorch3d.loss import chamfer_distance
# from torch.nn.utils.rnn import pad_sequence
# from equi_pers.equi2pers_v3 import equi2pers
# from thop import profile

parser = argparse.ArgumentParser(description='360Transformer')
parser.add_argument('--input_dir', default='/home/ps/data/dataset/360Image/Dataset/Matterport3D/',
                    help='input data directory')

parser.add_argument('--testfile', default='./filenames/matterport3d_test.txt',
                    help='validation file name')

parser.add_argument('--batch', type=int, default=1,
                    help='number of batch to train')

parser.add_argument('--patchsize', type=list, default=(256, 256),
                    help='patch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--fov', type=float, default=80,
                    help='field of view')
parser.add_argument('--iter', type=int, default=2,
                    help='number of iterations')
parser.add_argument('--nrows', type=int, default=4,
                    help='number of rows, options are 3, 4, 5, 6')

parser.add_argument('--checkpoint', default="/home/ps/data/haoai/HDRFuse_github/results/matterport3D/hrdfuse_256_80/",#"/hpc/users/CONNECT/haoai/HRDFuse/results/hrdfuse_128_80_hyy",
                    help='load checkpoint path')

parser.add_argument('--save_path', default='./compare/matterport3d_test_256_80_1/',
                    help='save checkpoint path')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--min_val', type=float, default=0.1,
                    help='number of batch to train')
parser.add_argument('--max_val', type=float, default=10.0,
                    help='number of batch to train')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
else:
    shutil.rmtree(args.save_path)
result_view_dir = args.save_path
# Random Seed -----------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# ------------------------------------------tensorboard_pathf training files
input_dir = args.input_dir
val_file_list = args.testfile  # File with list of validation files
# ------------------------------------
# -------------------------------------------------------------------
batch_size = args.batch

init_lr = args.lr
fov = (args.fov, args.fov)  # (48, 48)
patch_size = args.patchsize
nrows = args.nrows
npatches_dict = {3: 10, 4: 18, 5: 26, 6: 46}
min_val = args.min_val
max_val = args.max_val
iters = args.iter
# -------------------------------------------------------------------
# data loaders

val_dataloader = torch.utils.data.DataLoader(
    dataset=Dataset(
        root_path=input_dir,
        path_to_img_list=val_file_list),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False)

# ----------------------------------------------------------
# first network, coarse depth estimation
# option 1, resnet 360
num_gpu = torch.cuda.device_count()
network = spherical_fusion(nrows=nrows, npatches=npatches_dict[nrows], patch_size=patch_size, fov=fov, min_val=min_val,
                           max_val=max_val)
from thop import profile
# flops, params = profile(network, inputs=(torch.randn(1, 3, 512,1024)))
# print('FLOPs = ' + str(flops/(1000**3)) + 'G')
# print('Params = ' + str(params/(1000**2)) + 'M')

# network = convert_model(network)
# network = spherical_fusion(nrows=nrows, npatches=npatches_dict[nrows], patch_size=patch_size, fov=fov)
network = convert_model(network)
# parallel on multi gpu

network = nn.DataParallel(network)
network.cuda()

# load pre-trained model
if args.checkpoint is not None:
    print("loading model from folder {}".format(args.checkpoint))
    if os.path.isfile(args.checkpoint):
        path = args.checkpoint
    else:
        path = os.path.join(args.checkpoint, "{}.tar".format("checkpoint_best"))
    model_dict = network.state_dict()
    pretrained_dict = torch.load(path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)

# ----------------------------------------------------------

print('## Batch size: {}'.format(batch_size))
print('## learning rate: {}'.format(init_lr))
print('## patch size:', patch_size)
print('## fov:', args.fov)
print('## Number of first model parameters: {}'.format(
    sum([p.data.nelement() for p in network.parameters() if p.requires_grad is True])))
# --------------------------------------------------

# Optimizer ----------
optimizer = optim.AdamW(list(network.parameters()),
                        lr=init_lr, betas=(0.9, 0.999), weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {'val': self.val,
                'sum': self.sum,
                'count': self.count,
                'avg': self.avg}

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


abs_rel_error_meter = AverageMeter()
sq_rel_error_meter = AverageMeter()
lin_rms_sq_error_meter = AverageMeter()
log_rms_sq_error_meter = AverageMeter()
d1_inlier_meter = AverageMeter()
d2_inlier_meter = AverageMeter()
d3_inlier_meter = AverageMeter()

def compute_eval_metrics(output, gt, depth_mask):
    '''
    Computes metrics used to evaluate the model
    '''
    depth_pred = output
    gt_depth = gt

    N = output.shape[0]#depth_mask.sum()

    # Align the prediction scales via median

    median_scaling_factor = gt_depth[depth_mask > 0].median() / depth_pred[depth_mask > 0].median()
    depth_pred *= median_scaling_factor

    abs_rel = abs_rel_error(depth_pred, gt_depth, depth_mask)
    sq_rel = sq_rel_error(depth_pred, gt_depth, depth_mask)
    rms_sq_lin = torch.sqrt(lin_rms_sq_error(depth_pred, gt_depth, depth_mask))
    rms_sq_log = torch.sqrt(log_rms_sq_error(depth_pred, gt_depth, depth_mask))
    d1 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=1)
    d2 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=2)
    d3 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=3)

    abs_rel_error_meter.update(abs_rel, N)
    sq_rel_error_meter.update(sq_rel, N)
    lin_rms_sq_error_meter.update(rms_sq_lin, N)
    log_rms_sq_error_meter.update(rms_sq_log, N)
    d1_inlier_meter.update(d1, N)
    d2_inlier_meter.update(d2, N)
    d3_inlier_meter.update(d3, N)


# Main Function ---------------------------------------------------------------------------------------------
def main():
    global_step = 0
    global_val = 0
    print('-------------Validate-----------')
    network.eval()
    index_k = 0

    for batch_idx, (rgb, depth, mask) in tqdm(enumerate(val_dataloader)):
        bs, _, h, w = rgb.shape

        rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()

        with torch.no_grad():
            # equi_outputs_list = network(rgb, iter=iters)
            # equi_depth_outputs = network( rgb)
            # equi_depth_outputs, bin_edges_erp = network(rgb, index_k=index_k)

            equi_depth_outputs, bin_edges_erp, local_depth_outputs, global_depth_outputs, _, _ = network(rgb
                                                                                                    )
            # _, similarity_index_map = torch.max(similarity_map, dim=1, keepdim=True)
            equi_depth_outputs = equi_depth_outputs.detach() * mask
            # equi_depth_outputs = equi_outputs_list[-1]
            error = torch.abs(depth - equi_depth_outputs) * mask
            error[error < 0.1] = 0

        # tangent_img, _, _, _ = equi2pers(rgb, fov=fov, nrows=nrows, patch_size=patch_size)
        rgb_img = rgb.detach().cpu().numpy()
        # tangent_img = tangent_img.detach().cpu().numpy()
        # bin_edges_erp= bin_edges_erp.detach().cpu().numpy()
        # bin_edges_tangent = bin_edges_tangent.flatten(2).detach().cpu().numpy()

        depth_prediction = equi_depth_outputs.detach().cpu().numpy()
        local_depth_prediction = local_depth_outputs.detach().cpu().numpy()

        # similarity_index_map = np.reshape(similarity_index_map,[1,-1])
        # counts={}
        # for i in similarity_index_map[0]:
        #
        #     counts[i]=counts.get(i,0)+1
        # print(counts)
        # import sys
        # sys.exit()
        # global_depth_prediction = global_depth_outputs.detach().cpu().numpy()

        equi_gt = depth.detach().cpu().numpy()
        error_img = error.detach().cpu().numpy()
        depth_prediction[depth_prediction > max_val] = 0
        local_depth_prediction[local_depth_prediction > 10] = 0
        # global_depth_prediction[global_depth_prediction > 10] = 0

        # save raw 3D point cloud reconstruction as ply file
        coords = np.stack(np.meshgrid(range(w), range(h)), -1)
        coords = np.reshape(coords, [-1, 2])
        coords += 1
        uv = coords2uv(coords, w, h)
        xyz = uv2xyz(uv)
        xyz = torch.from_numpy(xyz).to(rgb.device)
        xyz = xyz.unsqueeze(0).repeat(bs, 1, 1)
        gtxyz = xyz * depth.reshape(bs, w * h, 1)
        predxyz = xyz * equi_depth_outputs.reshape(bs, w * h, 1)
        gtxyz = gtxyz.detach().cpu().numpy()
        predxyz = predxyz.detach().cpu().numpy()
        error = error.detach().cpu().numpy()

        # equi_mask *= mask
        compute_eval_metrics(equi_depth_outputs, depth, mask)

        rmse = torch.sqrt(lin_rms_sq_error(equi_depth_outputs, depth, mask))
        if rmse < 0.25 :
            batch_result_view_dir = os.path.join(result_view_dir, str(batch_idx))
            if not os.path.isdir(batch_result_view_dir):
                os.makedirs(batch_result_view_dir)
            else:
                shutil.rmtree(batch_result_view_dir)
            rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)
            depth_pred_img = depth_prediction[0, 0, :, :]
            bin_edges_erp = bin_edges_erp[0,:]

            local_pred_img = local_depth_prediction[0, 0, :, :]
            # global_pred_img = global_depth_prediction[0, 0, :, :]
            depth_gt_img = equi_gt[0, 0, :, :]
            error_img = error_img[0, 0, :, :]
            gtxyz_np = gtxyz[0, ...]
            predxyz_np = predxyz[0, ...]

            cv2.imwrite('{}/test_equi_rgb_{}_{}.png'.format(batch_result_view_dir, batch_idx,rmse),
                        rgb_img * 255)
            plot.imsave('{}/test_equi_depth_pred_{}.png'.format(batch_result_view_dir, batch_idx),
                        depth_pred_img, cmap="jet")
            plot.imsave('{}/test_local_depth_pred_{}.png'.format(batch_result_view_dir, batch_idx),
                        local_pred_img, cmap="jet")
            # plot.imsave('{}/test_global_depth_pred_{}.png'.format(batch_result_view_dir, batch_idx),
            #             global_pred_img, cmap="jet")
            plot.imsave('{}/test_equi_gt_{}.png'.format(batch_result_view_dir, batch_idx),
                        depth_gt_img, cmap="jet")
            plot.imsave('{}/test_error_{}.png'.format(batch_result_view_dir, batch_idx),
                        error_img, cmap="jet")
            rgb_img = np.reshape(rgb_img * 255, (-1, 3)).astype(np.uint8)
            # write_ply('{}/test_gt_{}'.format(batch_result_view_dir, batch_idx), [gtxyz_np, rgb_img],
            #           ['x', 'y', 'z', 'blue', 'green', 'red'])
            # write_ply('{}/test_pred_{}'.format(batch_result_view_dir, batch_idx), [predxyz_np, rgb_img],
            #           ['x', 'y', 'z', 'blue', 'green', 'red'])

        global_val += 1
        # ------------
    print(
        '  Avg. Abs. Rel. Error: {:.4f}\n'
        '  Avg. Sq. Rel. Error: {:.4f}\n'
        '  Avg. Lin. RMS Error: {:.4f}\n'
        '  Avg. Log RMS Error: {:.4f}\n'
        '  Inlier D1: {:.4f}\n'
        '  Inlier D2: {:.4f}\n'
        '  Inlier D3: {:.4f}\n'.format(
            abs_rel_error_meter.avg,
            sq_rel_error_meter.avg,
            lin_rms_sq_error_meter.avg,
            log_rms_sq_error_meter.avg,
            d1_inlier_meter.avg,
            d2_inlier_meter.avg,
            d3_inlier_meter.avg,
            ))
# ---------------------------------------------------------------------------------

if __name__ == '__main__':
    main()


