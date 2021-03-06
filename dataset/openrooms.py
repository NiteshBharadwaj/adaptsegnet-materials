import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import glob
import os.path as osp
import random
from tqdm import tqdm
import time
import cv2
import struct

from PIL import Image

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class OpenRoomsSegmentation(data.Dataset):
    """`OpenRooms _ Segmentation Dataset.
    Args:
        root (string): Root directory of the OpenRooms Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()
    def __init__(self, root, opt, split,  transform=None, randomize=True, loadNeighborImage=False,
                 load_semantics=False,
                 load_boundary=False, dirs=['main_xml', 'main_xml1',
                                            'mainDiffLight_xml', 'mainDiffLight_xml1',
                                            'mainDiffMat_xml', 'mainDiffMat_xml1'],
                 imHeight=240, imWidth=320,
                 phase='TRAIN', rseed=None, cascadeLevel=0,
                 isLight=False, isAllLight=False,
                 envHeight=8, envWidth=16, envRow=120, envCol=160,
                 SGNum=12, labelLevel=1,
                 remap_labels = None):
        self.root = root
        self.options = opt
        self.split = split
        self.random = randomize
        self.dataset_name = 'openrooms-' + root
        assert self.split in ['train', 'val', 'test']
        self.sceneFile = split +  ".txt"
        if opt.train_subset_id!=-1 and split=="train":
            self.sceneFile = split + "_"+str(opt.train_subset_id)+"k.txt"
        print("Readin from "+ self.sceneFile)
        with open(self.sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        self.sceneList = [x.strip() for x in sceneList]
        #self.sceneList = self.sceneList[0:100]
        num_scenes = len(sceneList)
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        self.cascadeLevel = cascadeLevel
        self.isLight = isLight
        self.isAllLight = isAllLight
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.envRow = envRow
        self.envCol = envCol
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.SGNum = SGNum
        self.remap_labels = remap_labels

        self.transform = transform
        
        print('Scene num for split %s: %d; total scenes: %d' % (self.split, len(self.sceneList), num_scenes))
        # Permute the image list
        self.count = len(self.sceneList)
        self.perm = list(range(self.count))

        if self.random:
            if rseed is not None:
                random.seed(0)
            else:
                t = int(time.time() * 1000000)
                np.random.seed(((t & 0xff000000) >> 24) +
                           ((t & 0x00ff0000) >> 8) +
                           ((t & 0x0000ff00) << 8) +
                           ((t & 0x000000ff) << 24))
            random.shuffle(self.perm)
        
        # Nitesh
        # We want to work on rgb images here
        self.if_hdr = False
        self.invalid_index = 255
        
        light_to_window_dict = {0:31}
        light_to_window = lambda x: light_to_window_dict.get(x,x)
        self.light_to_window = np.vectorize(light_to_window)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))

        if self.random:
            index = np.random.randint(len(self.perm))
        else:
            index = index % len(self.perm)
            pass
        # Read segmentation
        # print(segArea.shape, segEnv.shape, segObj.shape)
        # print(segObj.shape)

        # Read Image
        scene = self.sceneList[self.perm[index]].split(" ")

        hdr_file = osp.join(self.root,scene[0].strip())
        mask_file = osp.join(self.root,scene[1].strip().replace("imsemLabel2_","imsemLabel_").replace("mainDiffLight", "main").replace("mainDiffMat", "main"))
        #print(mask_file)
        # if self.opt.if_hdr:
        im = self.loadHdr(hdr_file)
        seg = np.ones((1, im.shape[1], im.shape[2])) 
        # Random scale the image
        im, scale = self.scaleHdr(im, seg)
        if not self.if_hdr:
            im_not_hdr = np.clip(im ** (1.0 / 2.2), 0., 1.)
            image = (255. * im_not_hdr.transpose(1, 2, 0)).astype(np.uint8)
            #if self.transform is not None:
            #    image = Image.fromarray(image)
        if image is None:
            print(hdr_file)
        mask = self.loadNPY(mask_file)
        mask_labels = mask.astype(np.uint8)
        mask_labels = self.light_to_window(mask_labels)
        if self.remap_labels is not None:
            mask_labels = self.remap_labels(mask_labels)
        target = mask_labels
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target, hdr_file


    def __len__(self):
        return len(self.perm)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[(mask+1)%256]

    def loadHdr(self, imName):
        if not (osp.isfile(imName)):
            print(imName)
            assert (False)
        im = cv2.imread(imName, -1)
        # print(imName, im.shape, im.dtype)

        if im is None:
            print(imName)
            assert (False)
        im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA)
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]
        return im

    def scaleHdr(self, hdr, seg):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.split=="train":
            scale = (0.95 - 0.1 * np.random.random()) / np.clip(
                intensityArr[int(0.95 * self.imWidth * self.imHeight * 3)], 0.1, None)
        else:
            scale = (0.95 - 0.05) / np.clip(intensityArr[int(0.95 * self.imWidth * self.imHeight * 3)], 0.1, None)
        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale

    def loadImage(self, imName, isGama=False):
        if not (osp.isfile(imName)):
            print(imName)
            assert (False)

        im = Image.open(imName)
        im = im.resize([self.imWidth, self.imHeight], Image.ANTIALIAS)

        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])

        return im
    def loadBinary(self, imName, channels=1, dtype=np.float32, if_resize=True):
        assert dtype in [np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
        if not (osp.isfile(imName)):
            print(imName)
            assert (False)
        with open(imName, 'rb') as fIn:
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            dBuffer = fIn.read(4 * channels * width * height)
            if dtype == np.float32:
                decode_char = 'f'
            elif dtype == np.int32:
                decode_char = 'i'
            depth = np.asarray(struct.unpack(decode_char * channels * height * width, dBuffer), dtype=dtype)
            depth = depth.reshape([height, width, channels])
            if if_resize:
                # print(self.imWidth, self.imHeight, width, height)
                if dtype == np.float32:
                    depth = cv2.resize(depth, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA)
                elif dtype == np.int32:
                    depth = cv2.resize(depth.astype(np.float32), (self.imWidth, self.imHeight),
                                       interpolation=cv2.INTER_NEAREST)
                    depth = depth.astype(np.int32)

            depth = np.squeeze(depth)

        return depth[np.newaxis, :, :]

    def loadNPY(self, imName, dtype=np.int32, if_resize=True):
        depth = np.load(imName)
        if if_resize:
            # print(self.imWidth, self.imHeight, width, height)
            if dtype == np.float32:
                depth = cv2.resize(
                    depth, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA)
            elif dtype == np.int32:
                depth = cv2.resize(depth.astype(
                    np.float32), (self.imWidth, self.imHeight), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.int32)

        depth = np.squeeze(depth)

        return depth
