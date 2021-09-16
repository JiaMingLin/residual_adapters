# pretrain at cifar100, 
# then serial run through remaining dataset, 
#  - record the max acc_val
#  - save the corresponding model


pretrain_ds = 'cifar100'
target_ds = ['aircraft', 'daimlerpedcls', 'dtd', 'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers']

GPU_cores = [0,1,2,3]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import models
import os
import time
import argparse
import numpy as np

from torch.autograd import Variable

import imdbfolder_coco as imdbfolder
import config_task
import utils_pytorch
import train_adapter as ta
import train_from_scratch as ts

parser = argparse.ArgumentParser(description='PyTorch Residual Adapters training')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1., type=float, help='weight decay for the classification layer')
parser.add_argument('--wd3x3', default=1., type=float, nargs='+', help='weight decay for the 3x3')
parser.add_argument('--wd1x1', default=1., type=float, nargs='+', help='weight decay for the 1x1')
parser.add_argument('--nb_epochs', default=120, type=int, help='nb epochs')
parser.add_argument('--step1', default=80, type=int, help='nb epochs before first lr decrease')
parser.add_argument('--step2', default=100, type=int, help='nb epochs before second lr decrease')
parser.add_argument('--mode', default='parallel_adapters', type=str, help='Task adaptation mode')
parser.add_argument('--proj', default='11', type=str, help='Position of the adaptation module')
parser.add_argument('--dropout', default='00', type=str, help='Position of dropouts')
parser.add_argument('--expdir', default='/data/mdl/exp/adapter/', help='Save folder')
parser.add_argument('--datadir', default='/data/mdl/decathlon/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='/data/mdl/decathlon/annotations/', help='annotation folder')
parser.add_argument('--source', default='', type=str, help='Network source')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--factor', default='1.', type=float, help='Width factor of the network')
parser.add_argument('--use_pretrain', default=1, type=bool, help='Using pretrain model(check --source specification)')
args = parser.parse_args()
args.archi ='default'
config_task.mode = args.mode
config_task.proj = args.proj
config_task.factor = args.factor
args.use_cuda = torch.cuda.is_available()
if type(args.dataset) is str:
    args.dataset = [args.dataset]

if type(args.wd3x3) is float:
    args.wd3x3 = [args.wd3x3]

if type(args.wd1x1) is float:
    args.wd1x1 = [args.wd1x1]

if not os.path.isdir(args.expdir):
    os.mkdir(args.expdir) 

config_task.decay3x3 = np.array(args.wd3x3) * 0.0001
config_task.decay1x1 = np.array(args.wd1x1) * 0.0001
args.wd = args.wd *  0.0001

args.ckpdir = args.expdir + '/checkpoint/'
args.svdir  = args.expdir + '/results/'

if not os.path.isdir(args.ckpdir):
    os.mkdir(args.ckpdir) 

if not os.path.isdir(args.svdir):
    os.mkdir(args.svdir) 

config_task.isdropout1 = (args.dropout[0] == '1')
config_task.isdropout2 = (args.dropout[1] == '1')

# train from scratch
if bool(args.use_pretrain) == False:
    args.dataset = pretrain_ds
    source_path = ts.train_val(args)
    args.source = source_path
else:
    if len(args.source) == 0:
        print("Using pre-trained model, but source path is not specified")
        exit(1)

for target in target_ds:
    args.dataset = target
    ta.train_val(args)

