import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import sys, os, pickle, time, datetime, re

from nn_models import * # contains ResNet and Logistic

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

num_epochs, batch_size = 30, 32

logdir = os.path.join("logs", "fixed_alpha-{}-beta-{}-seed-{}".format(args.alpha, args.beta, args.seed)) # directory for saving running log and plot
print(args)
print(logdir)
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir(logdir):
    os.mkdir(logdir)

use_cuda = torch.cuda.is_available()

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)

num_batches_per_epoch = len(train_dataset) // batch_size

# define model
input_dim, output_dim = 784, 10
net = LogisticRegression(input_dim, output_dim)

# define loss / criteria
criterion = nn.CrossEntropyLoss()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    net.to(device="cuda")
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# class labels
classes = tuple(range(10))

################################################################################
date_time_str =  re.sub("-| |:", "", str(datetime.datetime.now()).split('.')[0])

# running statistics and 
train_losses = [] # by mini-batches
train_accuracies = [] # anyway...
val_losses = [] # by epoch
val_accuracies = [] # by epoch

#################################################################################
# transform alpha, beta to pytorch lr and momentum
lr_pytorch, momentum_pytorch = args.alpha * (1 - args.beta), args.beta

optimizer = optim.SGD(net.parameters(), lr=lr_pytorch, momentum=momentum_pytorch, weight_decay=5e-4)
for epoch in range(1, num_epochs+1):
    net.train()
    running_loss = 0.0
    correct, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = Variable(inputs.view(-1, 28 * 28))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        # accuracy on training set
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        # back-propagation
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        mini_batch_loss = loss.item()
        train_losses.append(mini_batch_loss) # add to history
    # print()
    accu_train = correct / total
    train_accuracies.append(accu_train)
    ############################################################################
    # validate
    net.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            inputs = Variable(inputs.view(-1, 28*28))
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            mini_batch_loss_val = loss.item()
            val_loss += mini_batch_loss
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
    # compute val loss (ave. over all minibatches) and accuracy 
    accu_val = correct / total
    val_accuracies.append(accu_val)
    ave_train_loss = np.average(train_losses[-num_batches_per_epoch:])
    print("epoch {}, train loss (ave. over epoch) = {}, val accu. = {}".format(epoch, ave_train_loss, accu_val))
    np.savetxt(os.path.join(logdir, "train_losses.txt"), train_losses)
    np.savetxt(os.path.join(logdir, "val_accu.txt"), val_accuracies)