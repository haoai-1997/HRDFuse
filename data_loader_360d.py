import torch
import torch.utils.data
import numpy as np
# import OpenEXR, Imath, array
import scipy.io
import math
import os.path as osp
import cv2


def recover_filename(file_name):
    splits = file_name.split('.')
    rot_ang = splits[0].split('_')[-1]
    file_name = splits[0][:-len(rot_ang)] + "0." + splits[-2] + "." + splits[-1]

    return file_name, int(rot_ang)


def random_uniform(low, high, size):
    n = (high - low) * torch.rand(size) + low
    return n.numpy()

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list
class Dataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self,
                 root_path,
                 path_to_img_list,
                 rotate=False,
                 flip=False,
                 gamma=False):

        # Set up a reader to load the panos
        self.root_path = root_path

        # Create tuples of inputs/GT
        self.image_list = read_list(path_to_img_list)

        # Max depth for GT
        self.max_depth = 8.0
        self.min_depth = 0.1
        self.rotate = rotate
        self.flip = flip
        self.gamma = gamma
        self.pano_h = 256
        self.pano_w = 512

    def __getitem__(self, idx):
        '''Load the data'''

        # Select the panos to load
        relative_paths = self.image_list[idx]

        # Load the panos
        relative_basename = osp.splitext((relative_paths[0]))[0]
        basename = osp.splitext(osp.basename(relative_paths[0]))[0]
        rgb_name, rot_ang = recover_filename(self.root_path + relative_paths[0])
        rgb = self.readRGBPano(rgb_name)
        # depth = self.readDepthPano(self.root_path + relative_paths[3])
        depth_name, _ = recover_filename(self.root_path + relative_paths[1])
        depth = self.readDepthPano(depth_name)
        rgb = rgb.astype(np.float32) / 255

        # Random flip
        if self.flip:
            if torch.randint(2, size=(1,))[0].item() == 0:
                rgb = np.flip(rgb, axis=1)
                depth = np.flip(depth, axis=1)

        # Random horizontal rotate
        if self.rotate:
            dx = torch.randint(rgb.shape[1], size=(1,))[0].item()
            dx = dx // (rgb.shape[1] // 4) * (rgb.shape[1] // 4)
            rgb = np.roll(rgb, dx, axis=1)
            depth = np.roll(depth, dx, axis=1)

        # Random gamma augmentation
        if self.gamma:
            p = random_uniform(1, 2, size=(1,))[0]
            if torch.randint(2, size=(1,))[0].item() == 0:
                p = 1 / p
                rgb = rgb ** p

        depth = np.expand_dims(depth, 0)
        depth_mask = ((depth <= self.max_depth) & (depth > self.min_depth)).astype(np.uint8)

        # Threshold depths
        depth *= depth_mask
        # Convert to torch format
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()  # depth
        depth = torch.from_numpy(depth.copy()).float()
        depth_mask = torch.from_numpy(depth_mask)

        # Return the set of pano data
        return rgb, depth, depth_mask

    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)

    def readRGBPano(self, path):
        '''Read RGB and normalize to [0,1].'''

        rgb = cv2.imread(path)
        if rgb is None:
            print(path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb, (self.pano_w, self.pano_h), interpolation=cv2.INTER_AREA)

        return rgb

    def readDepthPano(self, path):

        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (self.pano_w, self.pano_h), interpolation=cv2.INTER_AREA)
        # depth = depth / 4000
        return depth

        # return self.read_exr(path)[...,0].astype(np.float32)

        # mat_content = np.load(path, allow_pickle=True)
        # depth_img = mat_content['depth']
        # return depth_img.astype(np.float32)

        # depth = cv2.imread(path, -1).astype(np.float32)
        # depth = cv2.resize(depth, (512, 256), interpolation = cv2.INTER_AREA)
        # depth = depth/65535*128
        # return depth

    # def read_exr(self, image_fpath):
    #     f = OpenEXR.InputFile( image_fpath )
    #     dw = f.header()['dataWindow']
    #     w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    #     im = np.empty( (h, w, 3) )
    #
    #     # Read in the EXR
    #     FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    #     channels = f.channels( ["R", "G", "B"], FLOAT )
    #     for i, channel in enumerate( channels ):
    #         im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
    #     return im