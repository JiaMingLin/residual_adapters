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

def train_val(args):
    # Prepare data loaders
    train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(args.dataset,args.datadir,args.imdbdir,True)
    args.num_classes = num_classes

    # Create the network
    net = models.resnet26(num_classes)


    start_epoch = 0
    best_acc = 0  # best test accuracy
    results = np.zeros((4,start_epoch+args.nb_epochs,len(args.num_classes)))
    all_tasks = range(len(args.dataset))
    np.random.seed(1993)

    if args.use_cuda:
        net.cuda()
        cudnn.benchmark = True


    args.criterion = nn.CrossEntropyLoss()
    optimizer = sgd.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)


    print("Start training")
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        training_tasks = utils_pytorch.adjust_learning_rate_and_learning_taks(optimizer, epoch, args)
        st_time = time.time()
    
        # Training and validation
        ckp_path = args.ckpdir+'/ckpt'+config_task.mode+args.archi+args.proj+''.join(args.dataset)+'.t7'
        train_acc, train_loss = utils_pytorch.train(epoch, train_loaders, training_tasks, net, args, optimizer)
        test_acc, test_loss, best_acc = utils_pytorch.test(epoch,val_loaders, all_tasks, net, best_acc, args, optimizer)
        
        # Record statistics
        for i in range(len(training_tasks)):
            current_task = training_tasks[i]
            results[0:2,epoch,current_task] = [train_loss[i],train_acc[i]]
        for i in all_tasks:
            results[2:4,epoch,i] = [test_loss[i],test_acc[i]]
        np.save(args.svdir+'/results_'+'adapt'+str(args.seed)+args.dropout+args.mode+args.proj+''.join(args.dataset)+'wd3x3_'+str(args.wd3x3)+'_wd1x1_'+str(args.wd1x1)+str(args.wd)+str(args.nb_epochs)+str(args.step1)+str(args.step2),results)
        print('Epoch lasted {0}'.format(time.time()-st_time))