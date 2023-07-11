import cv2
import numpy as np
import os.path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from torchvision.transforms import ToTensor

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import imresize, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

from skimage import io


@DATASET_REGISTRY.register()
class LightFieldH5Dataset(data.Dataset):

    def __init__(self, opt):
        super(LightFieldH5Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        lq_path = gt_path.replace(self.gt_folder, self.lq_folder)
        # img_gt = io.imread(gt_path).astype(np.float32) / 255.0
        # img_lq = io.imread(lq_path).astype(np.float32) / 255.0
        img_gt = io.imread(gt_path).astype(np.float32) / 255.0
        img_lq = io.imread(lq_path).astype(np.float32) / 255.0

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = ToTensor()(img_gt)
        img_lq = ToTensor()(img_lq)
        # img_lq = img_lq.permute(1, 0, 2)
        img_lq = img_lq.permute(1, 2, 0)
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
