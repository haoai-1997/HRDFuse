from __future__ import print_function
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import math
from metrics import *
from tqdm import tqdm
from data_loader_matterport3d import Dataset
import cv2
import supervision as L
import spherical as S360
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
from model.spherical_fusion import spherical_fusion
# from model.spherical_fusion import *
from ply import write_ply
import csv
from util import *
import shutil
import torchvision.utils as vutils
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='360Transformer')
parser.add_argument('--input_dir', default='/home/ps/data/dataset/360Image/Dataset/Matterport3D/',
                    # parser.add_argument('--input_dir', default='/home/rtx2/NeurIPS/spherical_mvs/data/omnidepth',
                    # parser.add_argument('--input_dir', default='/media/rtx2/DATA/Structured3D/',
                    help='input data directory')
parser.add_argument('--trainfile', default='./filenames/matterport3d_train.txt',
                    help='train file name')
parser.add_argument('--testfile', default='./filenames/matterport3d_test.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train')
parser.add_argument('--batch', type=int, default=6,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=20,
                    help='number of batch to train')
parser.add_argument('--patchsize', type=list, default=(256, 256),
                    help='patch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--fov', type=float, default=80,
                    help='field of view')
parser.add_argument('--nrows', type=int, default=4,
                    help='number of rows, options are 3, 4, 5, 6')
parser.add_argument('--confidence', action='store_true', default=True,
                    help='use confidence map or not')
parser.add_argument('--checkpoint', default=None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='checkpoints',
                    help='save checkpoint path')
parser.add_argument('--save_path', default='results/matterport/hrdfuse_256_80',
                    help='save checkpoint path')
parser.add_argument('--tensorboard_path', default='logs',
                    help='tensorboard path')
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

# Save Checkpoint -------------------------------------------------------------
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
else:
    shutil.rmtree(args.save_path)
if not os.path.isdir(os.path.join(args.save_path, args.save_checkpoint)):
    os.makedirs(os.path.join(args.save_path, args.save_checkpoint))

# result visualize Path -----------------------
writer_path = os.path.join(args.save_path, args.tensorboard_path)
image_path = os.path.join(args.save_path, "image")
if not os.path.isdir(writer_path):
    os.makedirs(writer_path)
if not os.path.isdir(image_path):
    os.makedirs(image_path)
writer = SummaryWriter(log_dir=writer_path)

result_view_dir = args.save_path
shutil.copy('train_fusion.py', result_view_dir)
shutil.copy('model/spherical_fusion.py', result_view_dir)
shutil.copy('model/ViT/miniViT.py', result_view_dir)
shutil.copy('model/ViT/layers.py', result_view_dir)
# shutil.copy('model/spherical_model_iterative.py', result_view_dir)
# if os.path.exists('grid'):
#    shutil.rmtree('grid')
# -----------------------------------------

# Random Seed -----------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# ------------------------------------------tensorboard_pathf training files
input_dir = args.input_dir
train_file_list = args.trainfile
val_file_list = args.testfile  # File with list of validation files
# ------------------------------------
# -------------------------------------------------------------------
batch_size = args.batch
visualize_interval = args.visualize_interval
init_lr = args.lr
fov = (args.fov, args.fov)  # (48, 48)
patch_size = args.patchsize
nrows = args.nrows
npatches_dict = {3: 10, 4: 18, 5: 26, 6: 46}
min_val=args.min_val
max_val=args.max_val
# -------------------------------------------------------------------
# data loaders
train_dataloader = torch.utils.data.DataLoader(
    dataset=Dataset(
        rotate=True,
        flip=True,
        root_path=input_dir,
        path_to_img_list=train_file_list),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True)

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
network = spherical_fusion(nrows=nrows, npatches=npatches_dict[nrows], patch_size=patch_size, fov=fov, min_val=min_val, max_val=max_val)
# network = convert_model(network)

# parallel on multi gpu
network = nn.DataParallel(network)
network.cuda()

# load pre-trained model
if args.checkpoint is not None:
    print("loading model from folder {}".format(args.checkpoint))

    path = os.path.join(args.checkpoint, "{}.tar".format("checkpoint_latest"))
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

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.2)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-6, last_epoch=-1)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)


# ---------------------
class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0 * delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss
class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

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

    N = depth_mask.sum()

    # Align the prediction scales via median

    median_scaling_factor = gt_depth[depth_mask > 0].median() / depth_pred[depth_mask > 0].median()
    depth_pred *= median_scaling_factor

    abs_rel = abs_rel_error(depth_pred, gt_depth, depth_mask)
    sq_rel = sq_rel_error(depth_pred, gt_depth, depth_mask)
    rms_sq_lin = lin_rms_sq_error(depth_pred, gt_depth, depth_mask)
    rms_sq_log = log_rms_sq_error(depth_pred, gt_depth, depth_mask)
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
    best_accuracy = 1
    # save the evaluation results into a csv file
    csv_filename = os.path.join(result_view_dir, 'logs/result_log.csv')
    fields = ['epoch', 'Abs Rel', 'Sq Rel', 'Lin RMSE', 'log RMSE', 'D1', 'D2', 'D3', 'lr']
    csvfile = open(csv_filename, 'w', newline='')
    with csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

        # Start Training ---------------------------------------------------------
    start_full_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        total_chamfer_loss = 0
        total_depth_loss = 0
        total_image_loss = 0
        # -------------------------------
        network.train()
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (rgb, depth, mask) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            bs, _, h, w = rgb.shape

            rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()

            equi_depth_outputs, bin_erp_edges, local_depth_outputs, global_depth_outputs, queries_tangent, queries_erp= network(rgb)

            # error map, clip at 0.1
            error = torch.abs(depth - equi_depth_outputs) * mask
            error[error < 0.1] = 0

            # attention_weights = torch.ones_like(mask, dtype=torch.float32, device=mask.device)
            bin_loss = BinsChamferLoss()
            criterion_ueff = SILogLoss()
            smooth_l1_loss = BerhuLoss()
            l1_loss=nn.L1Loss()
            depth_loss = smooth_l1_loss(equi_depth_outputs, depth, mask=mask.to(torch.bool))

            depth_loss1 = smooth_l1_loss(local_depth_outputs, depth, mask=mask.to(torch.bool))

            depth_loss2 = smooth_l1_loss(global_depth_outputs, depth, mask=mask.to(torch.bool))
            # depth_loss = smooth_l1_loss(equi_depth_outputs , depth , mask=mask.to(torch.bool))
            l_chamfer = bin_loss(bin_erp_edges, depth)


            # gt_normal = depth2normal_gpu(depth)
            # pred_normal = depth2normal_gpu(equi_outputs)
            # normal_loss = 1 - torch.mean(torch.sum((pred_normal * gt_normal * mask), dim=[1, 2, 3], keepdim=True) / mask.sum())
            # gt_grad = imgrad_yx(depth)
            # pred_grad = imgrad_yx(equi_outputs)
            # grad_loss = L.direct.calculate_l1_loss(pred_grad, gt_grad, mask)
            loss = depth_loss + 0.1 * l_chamfer #+ 0.5 *l_feature # + normal_loss * 0.2 + grad_loss * 0.05

            rgb_img = rgb.detach().cpu().numpy()
            depth_prediction = equi_depth_outputs.detach().cpu().numpy()
            local_depth_prediction = local_depth_outputs.detach().cpu().numpy()
            global_depth_prediction = global_depth_outputs.detach().cpu().numpy()

            equi_gt = depth.detach().cpu().numpy()
            error_img = error.detach().cpu().numpy()
            depth_prediction[depth_prediction > 10] = 0
            local_depth_prediction[local_depth_prediction > 10] = 0
            global_depth_prediction[global_depth_prediction > 10] = 0

            if batch_idx % 200 == 0:
                rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)
                depth_pred_img = depth_prediction[0, 0, :, :]
                local_pred_img = local_depth_prediction[0, 0, :, :]
                global_pred_img = global_depth_prediction[0, 0, :, :]
                depth_gt_img = equi_gt[0, 0, :, :]
                error_img = error_img[0, 0, :, :]

                cv2.imwrite('{}/test_equi_rgb_{}.png'.format(image_path, batch_idx),
                            rgb_img * 255)
                plot.imsave('{}/test_equi_depth_pred_{}.png'.format(image_path, batch_idx),
                            depth_pred_img, cmap="jet")
                plot.imsave('{}/test_local_depth_pred_{}.png'.format(image_path, batch_idx),
                            local_pred_img, cmap="jet")
                plot.imsave('{}/test_global_depth_pred_{}.png'.format(image_path, batch_idx),
                            global_pred_img, cmap="jet")
                plot.imsave('{}/test_equi_gt_{}.png'.format(image_path, batch_idx),
                            depth_gt_img, cmap="jet")
                plot.imsave('{}/test_error_{}.png'.format(image_path, batch_idx),
                            error_img, cmap="jet")

            if batch_idx % visualize_interval == 0:
                writer.add_image('RGB', vutils.make_grid(rgb[:2, [2, 1, 0], :, :].data, nrow=4, normalize=True),
                                 batch_idx)
                writer.add_image('depth gt', colorize(vutils.make_grid(depth[:2, ...].data, nrow=4, normalize=False)),
                                 batch_idx)
                writer.add_image('depth pred',
                                 colorize(vutils.make_grid(equi_depth_outputs[:2, ...].data, nrow=4, normalize=False)),
                                 batch_idx)
                writer.add_image('local pred',
                                 colorize(vutils.make_grid(local_depth_outputs[:2, ...].data, nrow=4, normalize=False)),
                                 batch_idx)
                writer.add_image('global pred',
                                 colorize(vutils.make_grid(global_depth_outputs[:2, ...].data, nrow=4, normalize=False)),
                                 batch_idx)
                writer.add_image('error', colorize(vutils.make_grid(error[:2, ...].data, nrow=4, normalize=False)),
                                 batch_idx)
                # writer.add_image('normal', vutils.make_grid(pred_normal[:2, ...].data, nrow=4, normalize=True), batch_idx)
                # writer.add_image('normal gt', vutils.make_grid(gt_normal[:2, ...].data, nrow=4, normalize=True), batch_idx)
                # writer.add_image('confidence mask', colorize(vutils.make_grid(weight[:8, ...].data, nrow=4, normalize=False)), batch_idx)
                # writer.add_image('weight', colorize(vutils.make_grid(zero_weight[:4, ...].data, nrow=4, normalize=False)), batch_idx)
                # writer.add_image('depth coarse', colorize(vutils.make_grid(coarse_outputs[:2, ...].data, nrow=4, normalize=False)), batch_idx)

            loss.backward()

            optimizer.step()
            # scheduler.step()
            total_train_loss += loss.item()
            total_depth_loss += depth_loss.item()
            total_chamfer_loss += l_chamfer.item()

            global_step += 1
            if batch_idx % visualize_interval == 0 and batch_idx > 0:

                print('[Epoch %d--Iter %d]depth loss %.4f' %
                      (epoch, batch_idx, total_depth_loss / (batch_idx + 1)))
                print('[Epoch %d--Iter %d]chamfer loss %.4f' %
                      (epoch, batch_idx, total_chamfer_loss / (batch_idx + 1)))
        print('lr for epoch ', epoch, ' ', optimizer.param_groups[0]['lr'])
        torch.save(network.state_dict(), os.path.join(args.save_path, args.save_checkpoint) + '/checkpoint_latest.tar')
        # -----------------------------------------------------------------------------
        scheduler.step()
        # Valid ----------------------------------------------------------------------------------------------------
        if (epoch-1) % 2 == 0:
            print('-------------Validate Epoch', epoch, '-----------')
            network.eval()
            for batch_idx, (rgb, depth, mask) in tqdm(enumerate(val_dataloader)):
                bs, _, h, w = rgb.shape
                rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()

                with torch.no_grad():
                    equi_depth_outputs,_,local_depth_outputs, global_depth_outputs,_,_ = network(rgb)
                    error = torch.abs(depth - equi_depth_outputs) * mask
                    error[error < 0.1] = 0

                rgb_img = rgb.detach().cpu().numpy()
                depth_prediction = equi_depth_outputs.detach().cpu().numpy()
                local_depth_prediction = local_depth_outputs.detach().cpu().numpy()
                global_depth_prediction = global_depth_outputs.detach().cpu().numpy()

                equi_gt = depth.detach().cpu().numpy()
                error_img = error.detach().cpu().numpy()
                depth_prediction[depth_prediction > 10] = 0
                local_depth_prediction[local_depth_prediction > 10] = 0
                global_depth_prediction[global_depth_prediction > 10] = 0

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
                # error = error.detach().cpu().numpy()
                if batch_idx % 400 == 0:
                    rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)
                    depth_pred_img = depth_prediction[0, 0, :, :]
                    local_pred_img = local_depth_prediction[0, 0, :, :]
                    global_pred_img = global_depth_prediction[0, 0, :, :]
                    depth_gt_img = equi_gt[0, 0, :, :]
                    error_img = error_img[0, 0, :, :]
                    gtxyz_np = predxyz[0, ...]
                    predxyz_np = predxyz[0, ...]
                    cv2.imwrite('{}/test_equi_rgb_{}.png'.format(image_path, batch_idx),
                                rgb_img * 255)
                    plot.imsave('{}/test_equi_depth_pred_{}.png'.format(image_path, batch_idx),
                                depth_pred_img, cmap="jet")
                    plot.imsave('{}/test_local_depth_pred_{}.png'.format(image_path, batch_idx),
                                local_pred_img, cmap="jet")
                    plot.imsave('{}/test_global_depth_pred_{}.png'.format(image_path, batch_idx),
                                global_pred_img, cmap="jet")
                    plot.imsave('{}/test_equi_gt_{}.png'.format(image_path, batch_idx),
                                depth_gt_img, cmap="jet")
                    plot.imsave('{}/test_error_{}.png'.format(image_path, batch_idx),
                                error_img, cmap="jet")
                    rgb_img = np.reshape(rgb_img * 255, (-1, 3)).astype(np.uint8)
                    write_ply('{}/test_gt_{}'.format(image_path, batch_idx), [gtxyz_np, rgb_img],
                              ['x', 'y', 'z', 'blue', 'green', 'red'])
                    write_ply('{}/test_pred_{}'.format(image_path, batch_idx), [predxyz_np, rgb_img],
                              ['x', 'y', 'z', 'blue', 'green', 'red'])
                # equi_mask *= mask
                compute_eval_metrics(equi_depth_outputs, depth, mask)

                global_val += 1
                # ------------
            print('Epoch: {}\n'
                  '  Avg. Abs. Rel. Error: {:.4f}\n'
                  '  Avg. Sq. Rel. Error: {:.4f}\n'
                  '  Avg. Lin. RMS Error: {:.4f}\n'
                  '  Avg. Log RMS Error: {:.4f}\n'
                  '  Inlier D1: {:.4f}\n'
                  '  Inlier D2: {:.4f}\n'
                  '  Inlier D3: {:.4f}\n\n'.format(
                epoch,
                abs_rel_error_meter.avg,
                sq_rel_error_meter.avg,
                math.sqrt(lin_rms_sq_error_meter.avg),
                math.sqrt(log_rms_sq_error_meter.avg),
                d1_inlier_meter.avg,
                d2_inlier_meter.avg,
                d3_inlier_meter.avg))
            if abs_rel_error_meter.avg.item() < best_accuracy:
                torch.save(network.state_dict(),
                           os.path.join(args.save_path, args.save_checkpoint) + '/checkpoint_best.tar')
                best_accuracy = abs_rel_error_meter.avg.item()
            row = [epoch, '{:.4f}'.format(abs_rel_error_meter.avg.item()),
                   '{:.4f}'.format(sq_rel_error_meter.avg.item()),
                   '{:.4f}'.format(torch.sqrt(lin_rms_sq_error_meter.avg).item()),
                   '{:.4f}'.format(torch.sqrt(log_rms_sq_error_meter.avg).item()),
                   '{:.4f}'.format(d1_inlier_meter.avg.item()),
                   '{:.4f}'.format(d2_inlier_meter.avg.item()),
                   '{:.4f}'.format(d3_inlier_meter.avg.item()),
                   '{:.8f}'.format(optimizer.param_groups[0]['lr'])]
            with open(csv_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row)
            writer.add_scalar('abs rel', abs_rel_error_meter.avg, epoch)
            writer.add_scalar('log rmse', math.sqrt(log_rms_sq_error_meter.avg), epoch)
            abs_rel_error_meter.reset()
            sq_rel_error_meter.reset()
            lin_rms_sq_error_meter.reset()
            log_rms_sq_error_meter.reset()
            d1_inlier_meter.reset()
            d2_inlier_meter.reset()
            d3_inlier_meter.reset()
    # End Training
    print("Training Ended")
    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    writer.close()


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
