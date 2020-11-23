import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
from utils import transforms

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.openrooms import OpenRoomsSegmentation
from dataset.nyu_raw import NYU
from dataset.nyu import NYU as NYU_Labelled

IMG_MEAN = [104.00698793, 116.66876762, 122.67891434]
IMG_MEAN_RGB = [122.67891434, 116.66876762, 104.00698793]

MODEL = 'DeepLab'
BATCH_SIZE = 6
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/siggraphasia20dataset/code/Routine/DatasetCreation/'
DATA_LIST_PATH = './train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '320,240'
DATA_DIRECTORY_TARGET = '/data/datasets/nyuv2-python-toolkit/NYUv2/'
DATA_LIST_PATH_TARGET = './'
INPUT_SIZE_TARGET = '320,240'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 40
NUM_STEPS = 250000
NUM_STEPS_STOP = 250000  # early stopping
POWER = 0.9
RANDOM_SEED = 0
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'nyu'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--train_subset_id", type=int, default=100, choices=[10,20,30,50,100])
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument("--mode", type=str, default="uda",
                        help="uda/sda/baseline/baseline_tar")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""
    or_nyu_dict = {0:255, 1: 16, 2:40, 3: 39, 4:7, 5:14, 6: 39, 7: 12, 8: 38, 9: 40, 
            10: 10, 11:6, 12: 40, 13: 39, 14: 39, 15: 40, 16: 18, 17: 40, 18: 4, 19: 40,
            20: 40, 21: 5, 22: 40, 23: 40, 24: 30, 25: 36, 26: 38, 27: 40, 28: 3, 29: 40,
            30: 40, 31: 9, 32: 38, 33: 40, 34: 40, 35: 40, 36: 34, 37: 37, 38:40, 39:40,
            40: 39, 41: 8, 42: 3, 43: 1, 44: 2, 45: 22}
    or_nyu_map = lambda x: or_nyu_dict.get(x,x) - 1
    or_nyu_map = np.vectorize(or_nyu_map)

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    args.or_nyu_map = or_nyu_map
    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        elif args.restore_from == "":
            saved_state_dict = None
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 40 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

    model.train()
    model.to(device)

    cudnn.benchmark = True
    if args.mode!="baseline" and args.mode!="baseline_tar":
        # init D
        model_D1 = FCDiscriminator(num_classes=args.num_classes).to(device)
        model_D2 = FCDiscriminator(num_classes=args.num_classes).to(device)

        model_D1.train()
        model_D1.to(device)

        model_D2.train()
        model_D2.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    scale_min = 0.5
    scale_max = 2.0
    rotate_min = -10
    rotate_max = 10
    ignore_label = 255
    value_scale = 255 
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    args.width = w
    args.height = h
    train_transform = transforms.Compose([
            # et.ExtResize( 512 ),
            transforms.RandScale([scale_min, scale_max]),
            transforms.RandRotate([rotate_min, rotate_max], padding=IMG_MEAN_RGB, ignore_label=ignore_label),
            transforms.RandomGaussianBlur(),
            transforms.RandomHorizontalFlip(),
            transforms.Crop([args.height+1, args.width + 1], crop_type='rand', padding=IMG_MEAN_RGB, ignore_label=ignore_label),
            #et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            #et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            #et.ExtRandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN,
                            std=[1, 1, 1]),
        ])

    val_transform = transforms.Compose([
	    # et.ExtResize( 512 ),
	    transforms.Crop([args.height+1, args.width+1], crop_type='center', padding=IMG_MEAN_RGB, ignore_label=ignore_label),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=IMG_MEAN,
	    	    std=[1, 1, 1]),
	])
    if args.mode!="baseline_tar":
        src_train_dst = OpenRoomsSegmentation(root=args.data_dir, opt=args,
                                   split='train', transform=train_transform,
                                   imWidth = args.width, imHeight = args.height, remap_labels = args.or_nyu_map)
    else:
        src_train_dst = NYU_Labelled(root=args.data_dir_target, opt=args,
                         split='train', transform=train_transform,
                         imWidth = args.width, imHeight = args.height, phase="TRAIN",
                         randomize = True)
    tar_train_dst = NYU(root=args.data_dir_target, opt=args,
			 split='train', transform=train_transform,
			 imWidth = args.width, imHeight = args.height, phase="TRAIN",
			 randomize = True, mode=args.mode)
    tar_val_dst = NYU(root=args.data_dir, opt=args,
			 split='val', transform=val_transform,
			 imWidth = args.width, imHeight = args.height, phase="TRAIN",
			 randomize = False)
    trainloader = data.DataLoader(src_train_dst,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(tar_train_dst,
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    if args.mode!="baseline" and args.mode!="baseline_tar":
        optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D1.zero_grad()

        optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size[1]+1, input_size[0]+1), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1]+1, input_size_target[0]+1), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_seg_value1_tar = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_seg_value2_tar = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        if args.mode!="baseline" and args.mode!="baseline_tar":
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            adjust_learning_rate_D(optimizer_D1, i_iter)
            adjust_learning_rate_D(optimizer_D2, i_iter)
        sample_src = None
        sample_tar = None
        sample_res_src = None
        sample_res_tar = None
        sample_gt_src = None
        sample_gt_tar = None
        for sub_i in range(args.iter_size):

            # train G
            if args.mode!="baseline" and args.mode!="baseline_tar":
                # don't accumulate grads in D
                for param in model_D1.parameters():
                    param.requires_grad = False

                for param in model_D2.parameters():
                    param.requires_grad = False

            # train with source
            try:
                _, batch = trainloader_iter.__next__()
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = trainloader_iter.__next__()
            images, labels, _ = batch
            sample_src = images.clone()
            sample_gt_src = labels.clone()
            
            images = images.to(device)
            labels = labels.long().to(device)

            pred1, pred2 = model(images)
            pred1 = interp(pred1)
            pred2 = interp(pred2)
            sample_pred_src = pred2.detach().cpu()

            loss_seg1 = seg_loss(pred1, labels)
            loss_seg2 = seg_loss(pred2, labels)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.item() / args.iter_size
            loss_seg_value2 += loss_seg2.item() / args.iter_size
            
            # train with target
            try:
                _, batch = targetloader_iter.__next__()
            except:
                targetloader_iter = enumerate(targetloader) 
                _, batch = targetloader_iter.__next__()
            images, tar_labels, _, labelled = batch
            n_labelled = labelled.sum().detach().item()
            batch_size = images.shape[0]
            sample_tar = images.clone()
            sample_gt_tar = tar_labels.clone()
            images = images.to(device)

            pred_target1, pred_target2 = model(images)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)
            #print("N_labelled {}".format(n_labelled))
            if args.mode=="sda" and n_labelled!=0:
                labelled = labelled.to(device)==1
                tar_labels = tar_labels.to(device)
                loss_seg1_tar = seg_loss(pred_target1[labelled], tar_labels[labelled])
                loss_seg2_tar = seg_loss(pred_target2[labelled], tar_labels[labelled])
                loss_tar_labelled = loss_seg2_tar + args.lambda_seg * loss_seg1_tar
                loss_tar_labelled = loss_tar_labelled/args.iter_size
                loss_seg_value1_tar += loss_seg1_tar.item() / args.iter_size 
                loss_seg_value2_tar += loss_seg2_tar.item() / args.iter_size
            else:
                loss_tar_labelled = torch.zeros(1, requires_grad=True).float().to(device)
            # proper normalization
            sample_pred_tar = pred_target2.detach().cpu()
            if args.mode!="baseline" and args.mode!="baseline_tar":
                D_out1 = model_D1(F.softmax(pred_target1))
                D_out2 = model_D2(F.softmax(pred_target2))

                loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

                loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

                loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
                loss = loss / args.iter_size + loss_tar_labelled
                #loss = loss_tar_labelled
                loss.backward()
                loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size
                loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size
            # train D

            # bring back requires_grad
                for param in model_D1.parameters():
                    param.requires_grad = True

                for param in model_D2.parameters():
                    param.requires_grad = True

                # train with source
                pred1 = pred1.detach()
                pred2 = pred2.detach()

                D_out1 = model_D1(F.softmax(pred1))
                D_out2 = model_D2(F.softmax(pred2))

                loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

                loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

                loss_D1 = loss_D1 / args.iter_size / 2
                loss_D2 = loss_D2 / args.iter_size / 2

                loss_D1.backward()
                loss_D2.backward()

                loss_D_value1 += loss_D1.item()
                loss_D_value2 += loss_D2.item()

                # train with target
                pred_target1 = pred_target1.detach()
                pred_target2 = pred_target2.detach()

                D_out1 = model_D1(F.softmax(pred_target1))
                D_out2 = model_D2(F.softmax(pred_target2))

                loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))

                loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))

                loss_D1 = loss_D1 / args.iter_size / 2
                loss_D2 = loss_D2 / args.iter_size / 2

                loss_D1.backward()
                loss_D2.backward()

                loss_D_value1 += loss_D1.item()
                loss_D_value2 += loss_D2.item()

        optimizer.step()
        if args.mode!="baseline" and args.mode!="baseline_tar":
            optimizer_D1.step()
            optimizer_D2.step()
        if args.tensorboard:
            scalar_info = {
                'loss_seg1': loss_seg_value1,
                'loss_seg2': loss_seg_value2,
                'loss_adv_target1': loss_adv_target_value1,
                'loss_adv_target2': loss_adv_target_value2,
                'loss_D1': loss_D_value1,
                'loss_D2': loss_D_value2,
                'loss_seg1_tar': loss_seg_value1_tar,
                'loss_seg2_tar': loss_seg_value2_tar,
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)
            if i_iter % 1000 ==0:
                img = sample_src.cpu()[:,[2,1,0],:,:] + torch.from_numpy(np.array(IMG_MEAN_RGB).reshape(1,3,1,1)).float()
                img = img.type(torch.uint8)
                writer.add_images("Src/Images",img, i_iter)
                label = tar_train_dst.decode_target(sample_gt_src).transpose(0,3,1,2)
                writer.add_images("Src/Labels", label, i_iter)
                preds = sample_pred_src.permute(0,2,3,1).cpu().numpy()
                preds = np.asarray(np.argmax(preds, axis=3), dtype=np.uint8)
                preds = tar_train_dst.decode_target(preds).transpose(0,3,1,2)
                writer.add_images("Src/Preds", preds, i_iter)
                

                tar_img = sample_tar.cpu()[:,[2,1,0],:,:] + torch.from_numpy(np.array(IMG_MEAN_RGB).reshape(1,3,1,1)).float()
                tar_img = tar_img.type(torch.uint8)
                writer.add_images("Tar/Images",tar_img, i_iter)
                tar_label = tar_train_dst.decode_target(sample_gt_tar).transpose(0,3,1,2)
                writer.add_images("Tar/Labels", tar_label, i_iter)
                tar_preds = sample_pred_tar.permute(0,2,3,1).cpu().numpy()
                tar_preds = np.asarray(np.argmax(tar_preds, axis=3), dtype=np.uint8)
                tar_preds = tar_train_dst.decode_target(tar_preds).transpose(0,3,1,2)
                writer.add_images("Tar/Preds", tar_preds, i_iter)
        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f} loss_seg1_tar={8:.3f} loss_seg2_tar={9:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, loss_seg_value1_tar, loss_seg_value2_tar))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'OR_' + str(args.num_steps_stop) + '.pth'))
            if args.mode!="baseline" and args.mode!="baseline_tar":
                torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'OR_' + str(args.num_steps_stop) + '_D1.pth'))
                torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'OR_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'OR_' + str(i_iter) + '.pth'))
            if args.mode!="baseline" and args.mode!="baseline_tar":
                torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'OR_' + str(i_iter) + '_D1.pth'))
                torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'OR_' + str(i_iter) + '_D2.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
