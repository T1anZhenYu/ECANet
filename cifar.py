'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models as models
from progress.bar import Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from models.eca_module import *

import gc 

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=32, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', default=0.1, type=float,
                    help='lr')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--NewBN_tpye', dest='NewBNtype', type=int,default=1,
                    help='0:BN;')  
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
print('current gpu:')
print(torch.cuda.current_device())
print("Let's use", torch.cuda.device_count(), "GPUs!\n")
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)


        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # print("loss:",loss)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size})|  Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size})|  Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, epoch,is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    
    print("***************************saving***************************")
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best == 1:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
def convert_layers(model, layer_type_old=nn.BatchNorm2d, layer_type_new=NewBN, **kwargs):
    conversion_count = 0
    # print(type(torch.nn.modules.batchnorm.BatchNorm2d))
    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], num_converted = convert_layers(module, \
            layer_type_old, layer_type_new, **kwargs)
            conversion_count += num_converted
        # print('name:',name,' module:',module," 1 type:",nn.BatchNorm2d," 2 type",type(module),\
        # " change?:",type(module) == nn.BatchNorm2d)
        if type(module) == layer_type_old:

            layer_old = module
            layer_new = layer_type_new(layer_old.num_features, **kwargs)

            model._modules[name] = layer_new
            conversion_count += 1

    return model, conversion_count
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def main():
    global best_acc
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    # print("args:",args)
    writer = SummaryWriter(log_dir=args.checkpoint)
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            block_name=args.block_name,
        )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)


    if args.NewBNtype == 1:
        convert_layers(model,layer_type_new=NewBN)
    if args.NewBNtype == 2:
        convert_layers(model,layer_type_old=eca_layer,layer_type_new=eca_layer_batchwise)

    print(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()
    model_param = []
    for name, params in model.module.named_parameters():
        # print("name:", name)
        model_param += [{'params': [params]}]
    # optimizer = optim.SGD(model_param, lr=args.lr*args.train_batch/256, momentum=args.momentum, \
    #                       weight_decay=args.weight_decay)
    optimizer = optim.SGD(model_param, lr=args.lr, momentum=args.momentum, \
                          weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # adjust_learning_rate(optimizer, start_epoch)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,last_epoch=start_epoch-1)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    total_param = list( model.named_parameters())

    for epoch in range(start_epoch, args.epochs):
        
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch,
                                      use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

                # writer.add_scalar(name,params[0],epoch)
        writer.add_scalars('acc',{"test acc":test_acc,"train acc":train_acc},epoch)
        writer.add_scalars('loss',{"test loss":test_loss,"train loss":train_loss},epoch)

        # append logger file
        logger.append([optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, epoch+1, is_best, checkpoint=args.checkpoint)
        adjust_learning_rate(optimizer, epoch)
        # scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
    logger.close()
    logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)
if __name__ == '__main__':
    main()