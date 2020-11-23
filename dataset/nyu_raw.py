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

import imageio
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

def getRawFilenames(split):
    root = "/eccv20dataset/NYURaw"
    fname =  osp.join(root,split+"Img.txt")
    print(fname)
    with open(fname,"r") as f:
        fnames = f.readlines()
        fnames = [fn.strip() for fn in fnames]
        return fnames

class NYU(data.Dataset):
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
                 imHeight=240, imWidth=320,
                 phase='TRAIN', rseed=None, mode="uda", nyu_13_map = None):
        self.options = opt
        self.split = split
        self.random = randomize
        img_ids = []
        #with open(self.sceneFile, "r") as f:
        #    lines = f.readlines()
        #    for line in lines:
        #        img_ids.append(int(line.strip()))
        self.dataset_name = 'nyu-' + root
        assert self.split in ['train', 'val', 'test']
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        self.transform = transform
        self.mode = mode
        print("Phase: "+ phase.lower())
        self.imList = glob.glob(osp.join(root,"image",phase.lower(),"*.png"))
        if phase=="TRAIN": 
            imListRaw = getRawFilenames(phase.lower())
            self.imListRaw = imListRaw
            if mode!="sda":
                self.imList+=imListRaw

        self.imList = sorted(self.imList)
        random.seed(0)
        random.shuffle(self.imList)
        self.imListRaw = sorted(self.imListRaw)
        random.shuffle(self.imListRaw)
        if phase.upper() == 'TRAIN':
            num_scenes = len(self.imList)
            train_count = int(num_scenes * 0.9)
            val_count = num_scenes - train_count
            num_scenes_raw = len(self.imListRaw)
            train_count_raw = int(num_scenes_raw * 0.9)
            val_count_raw = num_scenes_raw - train_count_raw
            if self.split == 'train':
                self.imList = self.imList[:-val_count]
                self.imListRaw = self.imListRaw[:-val_count_raw]
            if self.split == 'val':
                self.imList = self.imList[-val_count:]
                self.imListRaw = self.imListRaw[-val_count_raw:]


        #self.imList = [x for x in self.imListAll if int(x.split("/")[-1].split(".png")[0]) in img_ids]
        #self.imList = self.imList[:100]
        self.maskList = [x.replace('image', 'seg40') for x in self.imList]
        
        # Permute the image list
        self.count = len(self.imList)
        self.perm = list(range(self.count))
        self.count_raw = len(self.imListRaw)
        self.nyu_13_map = nyu_13_map
        self.perm_raw = list(range(self.count_raw))
        if mode!="sda":
           print("Loaded NYU {} set with {} images with mode {}".format(split,self.count, self.mode))
        else:
           print("Loaded NYU {} set with {} images raw and {} images labelled with mode {}".format(split,self.count_raw,self.count, self.mode))
        if rseed is not None:
            random.seed(0)
        else:
            t = int(time.time() * 1000000)
            np.random.seed(((t & 0xff000000) >> 24) +
                           ((t & 0x00ff0000) >> 8) +
                           ((t & 0x0000ff00) << 8) +
                           ((t & 0x000000ff) << 24))
        random.shuffle(self.perm)
        random.shuffle(self.perm_raw)

        import scipy.io as sio
        ss = sio.loadmat("/data/nyu/classMapping40.mat")
        segClassMap = ss['mapClass'][0,:].tolist()
        segClassMap ={k:v for k,v in enumerate(segClassMap)}
        self.trans40 = np.vectorize(lambda x : segClassMap.get(x, x))
        # Nitesh
        # We want to work on rgb images here
        self.if_hdr = False
        self.invalid_index = 255
    
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
        if np.random.random()<0.5 and self.mode=="sda":
            raw = True
        else: 
            raw = False
        # Read segmentation
        # print(segArea.shape, segEnv.shape, segObj.shape)
        # print(segObj.shape)

        # Read Image
        if not raw:
            fname = self.imList[self.perm[index]%self.count]
            mask_fname = self.maskList[self.perm[index]%self.count]
        else:
            fname = self.imListRaw[self.perm_raw[index]%self.count_raw]
            mask_fname = fname
        is_labelled = not raw and self.mode=="sda"
        # if self.opt.if_hdr:
        im = self.loadImage(fname)
        # Random scale the image
        mask = self.loadImageIo(mask_fname)
        mask_labels = np.array(mask)
        if mask_labels.ndim>2:
             mask_labels = np.zeros((mask.shape[0], mask.shape[1]),dtype=np.uint8)
        else: 
             mask_labels = self.nyu_13_map(mask_labels)
        #mask_labels = self.trans40(mask_labels)
        target = mask_labels
        #print(np.unique(mask))
        if self.transform is not None:
            image, target = self.transform(im, target)

        return image, target, fname, is_labelled


    def __len__(self):
        if self.mode=="sda":
            return len(self.perm_raw)
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
        im = np.array(im, dtype=np.uint8)
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

    def loadImageIo(self, imName, dtype=np.int32, if_resize=True):
        depth = imageio.imread(imName)
        depth = np.array(depth)
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
