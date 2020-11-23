import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

from metrics.stream_metrics import StreamSegMetrics
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.nyu import NYU
from collections import OrderedDict
import os
from PIL import Image
from utils import transforms

import torch.nn as nn
IMG_MEAN = [104.00698793,116.66876762,122.67891434]

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
   
    parser.add_argument("--width", type=int, default=513)
    parser.add_argument("--height", type=int, default=513)
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    nyu_nyu_dict = {11:255, 13:255, 15:255, 17:255, 19:255, 20:255, 21: 255, 23: 255, 
            24:255, 25:255, 26:255, 27:255, 28:255, 29:255, 31:255, 32:255, 33:255}
    nyu_nyu_map = lambda x: nyu_nyu_dict.get(x+1,x)
    nyu_nyu_map = np.vectorize(nyu_nyu_map)
    args.nyu_nyu_map = nyu_nyu_map
    nyu_13_dict = {10:255, 11:10, 12:255, 13:11, 14:255, 15:12, 16:255, 17:13, 18:255, 19:255, 20:255, 21:14,22:255, 23:255,
             24:255, 25:255, 26:255, 27:255, 28:255, 29:255, 30:255, 31:255, 32:255, 33:255,34:15, 35:16,
             36:255, 37:255, 38:255, 39:255}
    nyu_13_map = lambda x: nyu_13_dict.get(x,x)
    nyu_13_map = np.vectorize(nyu_13_map)
    args.nyu_13_map = nyu_13_map
    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    ### for running different versions of pytorch
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    ###
    model.load_state_dict(saved_state_dict)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)

    model.eval()

    metrics = StreamSegMetrics(args.num_classes)
    metrics_remap = StreamSegMetrics(args.num_classes)
    ignore_label = 255
    value_scale = 255 
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    val_transform = transforms.Compose([
	    # et.ExtResize( 512 ),
	    transforms.Crop([args.height+1, args.width+1], crop_type='center', padding=IMG_MEAN, ignore_label=ignore_label),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=IMG_MEAN,
	    	    std=[1, 1, 1]),
	])
    val_dst = NYU(root=args.data_dir, opt=args,
			 split='val', transform=val_transform,
			 imWidth = args.width, imHeight = args.height, phase="TEST",
			 randomize = False, nyu_13_map=args.nyu_13_map)
    print("Dset Length {}".format(len(val_dst)))
    testloader = data.DataLoader(val_dst,
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(args.height+1, args.width+1), mode='bilinear', align_corners=True)
    metrics.reset()
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, targets, name = batch
        image = image.to(device)
        print(index)
        if args.model == 'DeeplabMulti':
            output1, output2 = model(image)
            output = interp(output2).cpu().data[0].numpy()
        elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
            output = model(image)
            output = interp(output).cpu().data[0].numpy()
        targets = targets.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        preds = output[None,:,:]
        #input_ = image.cpu().numpy()[0].transpose(1,2,0) + np.array(IMG_MEAN)
        metrics.update(targets, preds)
        #targets = args.nyu_nyu_map(targets)
        #preds = args.nyu_nyu_map(preds)
        metrics_remap.update(targets,preds)
        #input_ = Image.fromarray(input_.astype(np.uint8))
        #output_col = colorize_mask(output)
        #output = Image.fromarray(output)
        
        #name = name[0].split('/')[-1]
        #input_.save('%s/%s' % (args.save, name))
        #output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))
    print(metrics.get_results())
    print(metrics_remap.get_results())


if __name__ == '__main__':
    main()
