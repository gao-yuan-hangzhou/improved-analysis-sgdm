'''Train CIFAR10 with PyTorch.
    Compare a multi-stage parameter schedule with YellowFin
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, re, datetime, time
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

import numpy as np
import sys
sys.path.append("tuner_utils")
from yellowfin import YFOptimizer
from debug_plot import plot_func

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate') # 0.0001
parser.add_argument('--mu', default=0.0, type=float, help='momentum')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--logdir', type=str, default="./logs/yf")
parser.add_argument('--opt_method', type=str, default="SGD")
parser.add_argument('--lr_thresh', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument("--schedule_id", type=int, default=1)
# parser.add_argument("--alpha_values", type=float, default=0.05)
# parser.add_argument("--beta_values", type=float, default=)
args = parser.parse_args()

total_num_epochs = 50
batches_per_epoch = 390

# to store the log file
date_time_str =  re.sub("-| |:", "", str(datetime.datetime.now()).split('.')[0])
# args.logdir = "logs/exper-1-fixed-lr/alpha-{}-beta-{}-seed-{}-datetime-{}".format(args.alpha, args.beta, args.seed, date_time_str)
args.logdir = "logs/yf-lr-{}-seed-{}".format(args.lr, args.seed)

# make dir if not there
if not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)

import logging
logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(args.logdir + "/num.log", mode='w'),
            logging.StreamHandler(),
        ],
    )

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Data
logging.info('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
# if args.resume:
# if False: # don't do it
#     # Load checkpoint.
#     logging.info('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.t7')
#     net = checkpoint['net']
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
# else:
logging.info('==> Building model..')
# a standard choice
net = ResNet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
logging.info("using YF!!!!!!!!!!")
optimizer = YFOptimizer(net.parameters(), lr=args.lr, mu=args.mu, weight_decay=5e-4)

# Training
def train(epoch, opt,
    loss_list,\
    local_curv_list,\
    max_curv_list,\
    min_curv_list,\
    lr_list,\
    lr_t_list,\
    mu_t_list,\
    dr_list,\
    mu_list,\
    dist_list,\
    grad_var_list,\
    lr_g_norm_list,\
    lr_g_norm_squared_list,\
    move_lr_g_norm_list,\
    move_lr_g_norm_squared_list,\
    lr_grad_norm_clamp_act_list,\
    fast_view_act_list):
    logging.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch == 151:
        if args.opt_method == "YF":
            optimizer.set_lr_factor(optimizer.get_lr_factor() / 10.0)
        else:
            for group in optimizer.param_groups:
                    group['lr'] /= 10.0

    count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        count += 1
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # loss_list.append(loss.data.item())
        # if args.opt_method == "YF":
        local_curv_list.append(opt._global_state['grad_norm_squared'] )
        max_curv_list.append(opt._h_max)
        min_curv_list.append(opt._h_min)
        lr_list.append(opt._lr)
        mu_list.append(opt._mu)
        dr_list.append((opt._h_max + 1e-6) / (opt._h_min + 1e-6))
        dist_list.append(opt._dist_to_opt)
        grad_var_list.append(opt._grad_var)
        lr_g_norm_list.append(opt._lr * np.sqrt(opt._global_state['grad_norm_squared'].cpu() ) )
        lr_g_norm_squared_list.append(opt._lr * opt._global_state['grad_norm_squared'] )
        move_lr_g_norm_list.append(opt._optimizer.param_groups[0]["lr"] * np.sqrt(opt._global_state['grad_norm_squared'].cpu() ) )
        move_lr_g_norm_squared_list.append(opt._optimizer.param_groups[0]["lr"] * opt._global_state['grad_norm_squared'] )
        lr_t_list.append(opt._lr_t)
        mu_t_list.append(opt._mu_t)

        # train loss append
        loss_list.append(loss.data.item())
        train_loss += loss.data.item()
        # validation
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

        #if args.opt_method == "YF":
        #    lr_list.append(optimizer._optimizer.param_groups[0]['lr'] )
        #    mu_list.append(optimizer._optimizer.param_groups[0]['momentum'] )
        # else:
        #     lr_list.append(optimizer.param_groups[0]['lr'] )
        #     mu_list.append(optimizer.param_groups[0]['momentum'] )
    
    # print("num of batches per epoch = {}".format(count))

    return loss_list,\
    local_curv_list,\
    max_curv_list,\
    min_curv_list,\
    lr_list,\
    lr_t_list,\
    mu_t_list,\
    dr_list,\
    mu_list,\
    dist_list,\
    grad_var_list,\
    lr_g_norm_list,\
    lr_g_norm_squared_list,\
    move_lr_g_norm_list,\
    move_lr_g_norm_squared_list,\
    lr_grad_norm_clamp_act_list,\
    fast_view_act_list

    #return loss_list, lr_list, mu_list
 
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | time elapsed (sec): %.3f'
            % (test_loss/(batch_idx+1), 100.0*float(correct)/float(total), correct, total, time.time() - begin))

    # Save checkpoint.
    acc = float(correct)/float(total) # acc = 100.*float(correct)/float(total)
    # if acc > best_acc:
    #     logging.info('Saving..')
    #     state = {'net': net.module if use_cuda else net, 'acc': acc, 'epoch': epoch}
    #     best_acc = acc
    logging.info("Test acc: " + str(acc))
    return acc

if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)

train_loss_list = []
test_acc_list = []

loss_list = []
local_curv_list = []
max_curv_list = []
min_curv_list = []
lr_g_norm_list = []
lr_list = []
lr_t_list = []
mu_t_list = []
dr_list = []
mu_list = []
dist_list = []
grad_var_list = []

lr_g_norm_list = []
lr_g_norm_squared_list = []

move_lr_g_norm_list = []
move_lr_g_norm_squared_list = []

lr_grad_norm_clamp_act_list = []
fast_view_act_list = []

# train!
times_per_epoch = []

for epoch in range(total_num_epochs):
    print("Using YF! epoch = {}".format(epoch))
    begin = time.time()
    
    loss_list, \
    local_curv_list,\
    max_curv_list,\
    min_curv_list,\
    lr_list,\
    lr_t_list,\
    mu_t_list,\
    dr_list,\
    mu_list,\
    dist_list,\
    grad_var_list,\
    lr_g_norm_list,\
    lr_g_norm_squared_list,\
    move_lr_g_norm_list,\
    move_lr_g_norm_squared_list,\
    lr_grad_norm_clamp_act_list,\
    fast_view_act_list = \
      train(epoch, optimizer,
      loss_list, \
      local_curv_list,\
      max_curv_list,\
      min_curv_list,\
      lr_list,\
      lr_t_list,\
      mu_t_list,\
      dr_list,\
      mu_list,\
      dist_list,\
      grad_var_list,\
      lr_g_norm_list,\
      lr_g_norm_squared_list,\
      move_lr_g_norm_list,\
      move_lr_g_norm_squared_list,\
      lr_grad_norm_clamp_act_list,\
      fast_view_act_list)


    train_loss_list = loss_list # this is all minibatch losses
    # print("len(loss_list) = {}".format(len(loss_list)))
    test_acc = test(epoch)
    test_acc_list.append(test_acc)

    #lr_list += lr_epoch
    #mu_list += mu_epoch
    
    time_per_epoch = time.time() - begin
    times_per_epoch.append(time_per_epoch)
    np.savetxt(os.path.join(args.logdir, "train_losses.txt"), train_loss_list) # list of training losses
    np.savetxt(os.path.join(args.logdir, "val_acc.txt"), test_acc_list) # list of validation accuracies
    np.savetxt(os.path.join(args.logdir, "lr_pytorch_list.txt"), lr_list) # list of (pytorch) lr
    np.savetxt(os.path.join(args.logdir, "momentum_pytorch_list.txt"), mu_list) # list of (pytorch) momentum
    np.savetxt(os.path.join(args.logdir, "times_per_epoch.txt"), times_per_epoch) # duration of each epoch