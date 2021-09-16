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

#####################################


def init_target_model(args):
    checkpoint = torch.load(args.source)
    net_old = checkpoint['net']

    # init model, mode is cached at `config_task.mode`
    net = models.resnet26(num_classes)
    store_data = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
            store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
            m.weight.data = store_data[element]
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    names = []

    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'bns.' in name:
            names.append(name)
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    # Special case to copy the weight for the BN layers when the target and source networks have not the same number of BNs
    # To check, if BN in new model is random or copy from previous?
    import re
    condition_bn = 'noproblem'
    if len(names) != 51 and args.mode == 'series_adapters':
        condition_bn ='bns.....conv'

    for id_task in range(len(num_classes)):
        element = 0
        for name, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d) and 'bns.'+str(id_task) in name and not re.search(condition_bn,name):
                m.weight.data = store_data[element].clone()
                m.bias.data = store_data_bias[element].clone()
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    #net.linears[0].weight.data = net_old.linears[0].weight.data
    #net.linears[0].bias.data = net_old.linears[0].bias.data

    del net_old
    return net

def train_val(args):

    # Prepare data loaders
    train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(args.dataset,args.datadir,args.imdbdir,True)
    args.num_classes = num_classes

    # Load checkpoint and initialize the networks with the weights of a pretrained network
    print('==> Resuming from checkpoint..')
    net = init_target_model(args);

    start_epoch = 0
    best_acc = 0  # best test accuracy
    results = np.zeros((4,start_epoch+args.nb_epochs,len(args.num_classes)))

    # args.dataset is now array type, for joint training
    all_tasks = range(len(args.dataset))
    np.random.seed(1993)

    if args.use_cuda:
        net.cuda()
        cudnn.benchmark = True


    # Freeze 3*3 convolution layers
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
            m.weight.requires_grad = False


    args.criterion = nn.CrossEntropyLoss()
    optimizer = sgd.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)


    print("Start training")
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        training_tasks = utils_pytorch.adjust_learning_rate_and_learning_taks(optimizer, epoch, args)
        st_time = time.time()
    
        # Training and validation
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

