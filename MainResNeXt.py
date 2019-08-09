from __future__ import division

""" 
Trains a ResNeXt Model on Melanoma dataset.

Code taken from Github
Original:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms


#import ResNeXt
import torch.nn as nn
from torch.nn import init
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class ResNeXtNet(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXtNet, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 3
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_features=64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.
        """

        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality, self.base_width, self.widen_factor))
            else:
                block.add_module(name_, ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width, self.widen_factor))

        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 64, 1)
        x = x.view(-1, self.stages[3])
        x = self.classifier(x)

        return x



use_cuda = False







# Own modules
#from ResNeXt import ResNeXtNet
from dataloader import DataManager

# Methods
def load_data():
    start_index = 0
    data_size = 500
    train_size = 0.9

    # load images and labels (actually, it is get_sebor because this set is smaller and therefore faster loaded, no time to load the whole set)
    images, labels = mymanager.reduce_data(mymanager.get_images(as_tensor=True), mymanager.get_labels(as_tensor=True),
                                           indexFrom=start_index, indexTo=start_index + data_size)
    labels = labels.type(torch.FloatTensor)

    # check if images and labels have similar size
    if len(images) != len(labels):
        raise Exception('Error: Images and labels does not have equal length')

    # shuffle images and labels
    images, labels = mymanager.shuffle(images, labels)

    # split data in train, test and validation subset
    X_train, y_train, X_test, y_test, X_val, y_val = mymanager.datasplit(images, labels, train_size=train_size,
                                                                         validation_size=0.1)

    return X_train, y_train, X_test, y_test, X_val, y_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument('--data_path', type=str, help='Root for the dataset.', default='./ResNeXt/')
    parser.add_argument('--dataset', type=str, choices=['melanoma', 'empty'], default='melanoma',
                        help='Choose between the datasets')

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=3, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./ResNeXt/', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=2, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=3, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

    # Acceleration
    parser.add_argument('--ngpu', type=int, default=0, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    # I/O
    parser.add_argument('--log', type=str, default='./ResNeXt/', help='Log folder.')
    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    args.epochs = args.epochs * 128 // args.batch_size
    args.schedule = [x * 128 // args.batch_size for x in args.schedule]

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'melanoma':

        # Path
        image_path = './data/processed'
        mask_path = './data/ISIC-2017_Training_Part1_GroundTruth'
        label_file = './data/ISIC-2017_Training_Part3_GroundTruth.csv'

        # Generate a DataManager
        mymanager = DataManager(image_path, mask_path, label_file)

        # Load the data
        X_train, y_train, X_test, y_test, X_val, y_val = load_data()
        y_train = mymanager.convert_labels(y_train)
        y_test = mymanager.convert_labels(y_test)

        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)

        # Number of labels
        nlabels = 3
    else:
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 100

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer
    net = ResNeXtNet(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)
    print(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    if args.ngpu > 0:
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):

            if use_cuda:
                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable((target.type(torch.LongTensor)).cuda())
            else:
                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target.type(torch.LongTensor))

            # forward
            output = net(data)

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(output, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.2 + float(loss.data[0]) * 0.8

            del data
            del target
            del loss
            print(batch_idx)

        state['train_loss'] = loss_avg

    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable((target.type(torch.LongTensor)).cuda())
            else:
                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target.type(torch.LongTensor))

            # forward
            output = net(data)
            loss = F.cross_entropy(output, torch.max(target, 1)[1])

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()

            # test loss average
            loss_avg += loss.data[0]

        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)


    # Main loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        if epoch in args.schedule:
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

        state['epoch'] = epoch
        train()
        test()
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            torch.save(net, './ResNeXt/model1.pt')
            torch.save(net.state_dict(), os.path.join(args.save, 'model.pytorch'))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best accuracy: %f" % best_accuracy)

    log.close()